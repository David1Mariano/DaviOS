import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from core.reasoning import ReasoningResult
from core.reasoning import ReasoningEngine
from memory.context_analyzer import ContextAnalyzer
from memory.emotion_analyzer import EmotionAnalyzer
from memory.memory import Memory
from memory.memory_fact import MemoryFact
from memory.memory_interpreter import MemoryInterpreter
from memory.memory_manager import MemoryManager


class _FaultInjectingCursor:
    def __init__(self, cursor):
        self._cursor = cursor
        self.fail_releases = 0
        self.fail_rollback_to = 0

    def execute(self, sql, parameters=()):
        normalized = sql.strip().upper()
        if normalized.startswith("RELEASE SAVEPOINT") and self.fail_releases:
            self.fail_releases -= 1
            raise sqlite3.OperationalError("injected savepoint release failure")
        if normalized.startswith("ROLLBACK TO SAVEPOINT") and self.fail_rollback_to:
            self.fail_rollback_to -= 1
            raise sqlite3.OperationalError("injected savepoint rollback failure")
        return self._cursor.execute(sql, parameters)

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _FaultInjectingConnection:
    def __init__(self, connection):
        self._connection = connection
        self.cursor_proxy = _FaultInjectingCursor(connection.cursor())
        self.fail_commits = 0
        self.fail_after_commits = 0
        self.fail_rollbacks = 0

    def cursor(self):
        return self.cursor_proxy

    def commit(self):
        if self.fail_commits:
            self.fail_commits -= 1
            raise sqlite3.OperationalError("injected commit failure")
        result = self._connection.commit()
        if self.fail_after_commits:
            self.fail_after_commits -= 1
            raise sqlite3.OperationalError("injected post-commit failure")
        return result

    def rollback(self):
        if self.fail_rollbacks:
            self.fail_rollbacks -= 1
            raise sqlite3.OperationalError("injected connection rollback failure")
        return self._connection.rollback()

    def __getattr__(self, name):
        return getattr(self._connection, name)


class MemoryOperationTests(unittest.TestCase):
    def setUp(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.manager = MemoryManager(db_path=self.db_path)
        self.reasoning = ReasoningResult(
            intent="memory",
            action="memory",
            confidence=1.0,
            target_module="Memory",
            reasoning_log="test",
        )

    def tearDown(self):
        self.manager.database.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def memory(self, content, facts):
        return Memory(
            content=content,
            memory_type="preference",
            importance=0.8,
            emotion="neutral",
            emotional_intensity=0.0,
            facts=facts,
        )

    def fact(self, target, relation, **kwargs):
        return MemoryFact(
            target=target,
            relation=relation,
            emotion=kwargs.pop("emotion", "happiness" if relation == "like" else "dislike"),
            emotional_intensity=kwargs.pop("emotional_intensity", 5),
            **kwargs,
        )

    def apply(self, operation, new_memory, existing=None):
        self.reasoning.memory_operation = operation
        return self.manager.apply_memory_operation(
            self.reasoning,
            new_memory,
            existing,
        )

    def create_like(self):
        memory = self.memory("agora eu gosto de pizza", [self.fact("pizza", "like")])
        result = self.apply("create", memory)
        self.assertEqual(result["action"], "create")
        return result["memory"]

    def inject_connection_failures(self):
        proxy = _FaultInjectingConnection(self.manager.database.connection)
        self.manager.database.connection = proxy
        self.manager.database.cursor = proxy.cursor_proxy
        return proxy

    def test_create_and_duplicate_preference(self):
        self.create_like()
        duplicate = self.memory("eu gosto de pizza", [self.fact("pizza", "like")])
        result = self.apply("create", duplicate)
        # Declaração idêntica é reforço, não criação (Parte 8 da spec).
        self.assertEqual(result["action"], "reinforce")
        self.assertEqual(len(self.manager.recall()), 1)

    def test_like_dislike_like_updates_same_fact(self):
        existing = self.create_like()
        dislike = self.memory(
            "agora eu nao gosto de pizza",
            [self.fact("pizza", "dislike", negation=True)],
        )
        updated = self.apply("update", dislike, existing)
        self.assertEqual(updated["action"], "update")
        self.assertEqual(updated["memory"].id, existing.id)
        self.assertEqual(updated["memory"].facts[0].relation, "dislike")
        self.assertTrue(updated["memory"].facts[0].negation)

        like = self.memory("agora eu gosto de pizza", [self.fact("pizza", "like")])
        final = self.apply("update", like, updated["memory"])
        self.assertEqual(final["memory"].id, existing.id)
        self.assertEqual(final["memory"].facts[0].relation, "like")
        self.assertFalse(final["memory"].facts[0].negation)
        self.assertEqual(len(self.manager.recall()), 1)

    def test_identical_update_is_ignore_without_new_history(self):
        existing = self.create_like()
        history_before = self.manager.database.get_fact_history(existing.facts[0].id)
        duplicate = self.memory("eu gosto de pizza", [self.fact("pizza", "like")])
        result = self.apply("update", duplicate, existing)
        self.assertEqual(result["action"], "ignore")
        self.assertEqual(
            len(self.manager.database.get_fact_history(existing.facts[0].id)),
            len(history_before),
        )

    def test_add_fact_to_existing_memory(self):
        existing = self.create_like()
        coffee = self.memory(
            "eu nao gosto de cafe",
            [self.fact("coffee", "dislike", negation=True)],
        )
        result = self.apply("add", coffee, existing)
        self.assertEqual(result["action"], "add")
        self.assertEqual(len(result["memory"].facts), 2)
        self.assertEqual(len(self.manager.recall()), 1)

    def test_add_batch_same_key_preserves_duplicate_and_conflict_policies(self):
        scenarios = (
            (
                "identical",
                [self.fact("coffee", "like"), self.fact("coffee", "like")],
                "ignore",
                1,
            ),
            (
                "contradictory",
                [
                    self.fact("coffee", "like"),
                    self.fact("coffee", "dislike", negation=True),
                ],
                "error",
                1,
            ),
            (
                "archived duplicates are distinct writes",
                [
                    self.fact("coffee", "like", status="archived"),
                    self.fact("coffee", "like", status="archived"),
                ],
                "add",
                3,
            ),
            (
                "conflicted duplicates are distinct writes",
                [
                    self.fact("coffee", "like", status="conflicted"),
                    self.fact("coffee", "like", status="conflicted"),
                ],
                "add",
                3,
            ),
        )

        for index_enabled in (False, True):
            for scenario, facts, expected_action, expected_count in scenarios:
                with self.subTest(
                    index_enabled=index_enabled,
                    scenario=scenario,
                ):
                    file_descriptor, db_path = tempfile.mkstemp(suffix=".db")
                    os.close(file_descriptor)
                    manager = MemoryManager(db_path=db_path)
                    try:
                        parent = self.memory(
                            "batch parent",
                            [self.fact("parent", "like")],
                        )
                        memory_id = manager.database.save_memory(parent)
                        existing = manager.database.get_memory(memory_id)
                        if index_enabled:
                            self.assertTrue(
                                manager.database.ensure_semantic_uniqueness_index()
                            )

                        incoming = self.memory("batch additions", facts)
                        result = manager.add_memory_fact(existing, incoming)

                        self.assertEqual(result["action"], expected_action)
                        persisted = manager.database.get_memory_facts(memory_id)
                        self.assertEqual(len(persisted), expected_count)
                        self.assertEqual(persisted[0].target, "parent")
                        self.assertFalse(manager.database.connection.in_transaction)
                    finally:
                        manager.database.close()
                        os.remove(db_path)

    def test_add_batch_detects_duplicate_or_conflict_at_third_position(self):
        scenarios = (
            (
                "third fact duplicates second",
                [
                    self.fact("tea", "like"),
                    self.fact("coffee", "like"),
                    self.fact("coffee", "like"),
                ],
                "ignore",
            ),
            (
                "third fact contradicts second",
                [
                    self.fact("tea", "like"),
                    self.fact("coffee", "like"),
                    self.fact("coffee", "dislike", negation=True),
                ],
                "error",
            ),
        )

        for scenario, facts, expected_action in scenarios:
            with self.subTest(scenario=scenario):
                file_descriptor, db_path = tempfile.mkstemp(suffix=".db")
                os.close(file_descriptor)
                manager = MemoryManager(db_path=db_path)
                try:
                    parent = self.memory(
                        "batch parent",
                        [self.fact("parent", "like")],
                    )
                    memory_id = manager.database.save_memory(parent)
                    existing = manager.database.get_memory(memory_id)
                    result = manager.add_memory_fact(
                        existing,
                        self.memory("third position batch", facts),
                    )

                    self.assertEqual(result["action"], expected_action)
                    persisted = manager.database.get_memory_facts(memory_id)
                    self.assertEqual([fact.target for fact in persisted], ["parent"])
                    self.assertFalse(manager.database.connection.in_transaction)
                finally:
                    manager.database.close()
                    os.remove(db_path)

    def test_add_batch_preserves_metadata_only_duplicate_policy(self):
        existing = self.memory(
            "batch parent",
            [self.fact("parent", "like")],
        )
        self.apply("create", existing)
        existing = self.manager.database.get_memory(existing.id)
        first = self.fact("coffee", "like", metadata={"origin": "first"})
        duplicate = self.fact("coffee", "like", metadata={"origin": "second"})

        result = self.manager.add_memory_fact(
            existing,
            self.memory("metadata-only difference", [first, duplicate]),
        )

        self.assertEqual(result["action"], "ignore")
        persisted = self.manager.database.get_memory_facts(existing.id)
        self.assertEqual([fact.target for fact in persisted], ["parent"])

    def test_add_batch_does_not_match_partial_target_similarity(self):
        existing = self.memory(
            "batch parent",
            [self.fact("parent", "like")],
        )
        self.apply("create", existing)
        existing = self.manager.database.get_memory(existing.id)

        result = self.manager.add_memory_fact(
            existing,
            self.memory(
                "partially similar targets",
                [
                    self.fact("coffee", "like"),
                    self.fact("coffee beans", "like"),
                ],
            ),
        )

        self.assertEqual(result["action"], "add")
        persisted = self.manager.database.get_memory_facts(existing.id)
        self.assertEqual(
            {fact.target for fact in persisted},
            {"parent", "coffee", "coffee beans"},
        )

    def test_add_batch_rolls_back_first_insert_when_second_insert_fails(self):
        existing = self.memory(
            "batch parent",
            [self.fact("parent", "like")],
        )
        self.apply("create", existing)
        existing = self.manager.database.get_memory(existing.id)
        incoming = self.memory(
            "two distinct additions",
            [self.fact("coffee", "like"), self.fact("tea", "like")],
        )
        database = self.manager.database
        original = database.add_memory_fact
        call_count = 0

        def fail_second(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("injected second add failure")
            return original(*args, **kwargs)

        database.add_memory_fact = fail_second
        try:
            with self.assertRaisesRegex(RuntimeError, "injected second add failure"):
                self.manager.add_memory_fact(existing, incoming)
        finally:
            database.add_memory_fact = original

        persisted = database.get_memory_facts(existing.id)
        self.assertEqual([fact.target for fact in persisted], ["parent"])
        self.assertFalse(database.connection.in_transaction)

    def test_missing_memory_operations_are_explicit_errors(self):
        memory = self.memory("pizza", [self.fact("pizza", "like")])
        self.assertEqual(
            self.apply("update", memory)["reason"],
            "update_without_existing_memory",
        )
        self.assertEqual(
            self.apply("add", memory)["reason"],
            "add_without_existing_memory",
        )
        self.assertEqual(len(self.manager.recall()), 0)

    def test_temporal_context_and_metadata_are_preserved(self):
        existing = self.create_like()
        existing_fact = existing.facts[0]
        existing_fact.importance = 0.91
        self.manager.database.update_memory_fact(
            existing.id,
            fact_id=existing_fact.id,
            importance=0.91,
            temporal_context="current",
            metadata={"origin": "test"},
        )
        update = self.memory(
            "agora eu nao gosto de pizza",
            [self.fact("pizza", "dislike", negation=True, temporal_context="current")],
        )
        result = self.apply("update", update, self.manager.database.get_memory(existing.id))
        fact = result["memory"].facts[0]
        self.assertEqual(fact.temporal_context, "current")
        self.assertEqual(fact.importance, 0.91)
        self.assertEqual(fact.metadata["origin"], "test")

    def test_history_contains_old_and_new_states(self):
        existing = self.create_like()
        update = self.memory(
            "agora eu nao gosto de pizza",
            [self.fact("pizza", "dislike", negation=True)],
        )
        self.apply("update", update, existing)
        history = self.manager.database.get_fact_history(existing.facts[0].id)
        self.assertEqual(history[0]["relation"], "like")
        self.assertEqual(history[-1]["relation"], "dislike")
        self.assertTrue(history[-1]["negation"])

    def test_legacy_active_fact_can_be_updated(self):
        existing = self.create_like()
        self.manager.database.cursor.execute(
            "UPDATE memory_facts SET source = ?, metadata = ? WHERE id = ?",
            ("legacy_migration", '{"legacy": true}', existing.facts[0].id),
        )
        self.manager.database.connection.commit()
        update = self.memory("eu nao gosto de pizza", [self.fact("pizza", "dislike", negation=True)])
        result = self.apply("update", update, self.manager.database.get_memory(existing.id))
        self.assertEqual(result["memory"].facts[0].relation, "dislike")
        self.assertEqual(result["memory"].facts[0].source, "user_statement")

    def test_rollback_on_update_failure(self):
        existing = self.create_like()
        original = self.manager.database.update_memory_fact

        def fail_once(*args, **kwargs):
            raise RuntimeError("forced failure")

        self.manager.database.update_memory_fact = fail_once
        update = self.memory("eu nao gosto de pizza", [self.fact("pizza", "dislike", negation=True)])
        with self.assertRaises(RuntimeError):
            self.apply("update", update, existing)
        self.manager.database.update_memory_fact = original
        persisted = self.manager.database.get_memory(existing.id)
        self.assertEqual(persisted.facts[0].relation, "like")
        self.assertFalse(persisted.facts[0].negation)

    def test_save_memory_rolls_back_when_fact_insert_fails(self):
        database = self.manager.database
        original = database.add_memory_fact

        def fail_insert(*args, **kwargs):
            raise RuntimeError("injected fact insert failure")

        database.add_memory_fact = fail_insert
        memory = self.memory("transaction probe", [self.fact("probe", "like")])
        try:
            with self.assertRaisesRegex(RuntimeError, "injected fact insert failure"):
                database.save_memory(memory)
        finally:
            database.add_memory_fact = original

        count = database.cursor.execute(
            "SELECT COUNT(*) FROM memories WHERE content = ?",
            ("transaction probe",),
        ).fetchone()[0]
        self.assertEqual(count, 0)
        self.assertFalse(database.connection.in_transaction)

    def test_isolated_write_commits_and_nested_write_waits_for_owner(self):
        database = self.manager.database
        isolated = self.memory("isolated write", [])

        database.save_memory(isolated)

        self.assertFalse(database.connection.in_transaction)
        self.assertIsNotNone(database.get_memory(isolated.id))

        nested = self.memory("nested write", [])
        with self.assertRaisesRegex(RuntimeError, "rollback outer transaction"):
            with database.atomic_write():
                database.save_memory(nested)
                self.assertTrue(database.connection.in_transaction)
                self.assertIsNotNone(database.get_memory(nested.id))
                raise RuntimeError("rollback outer transaction")

    def test_outer_transaction_rolls_back_successful_nested_write(self):
        database = self.manager.database
        nested = self.memory("nested rollback probe", [])

        with self.assertRaisesRegex(RuntimeError, "rollback outer transaction"):
            with database.atomic_write():
                database.save_memory(nested)
                self.assertTrue(database.connection.in_transaction)
                raise RuntimeError("rollback outer transaction")

        count = database.cursor.execute(
            "SELECT COUNT(*) FROM memories WHERE content = ?",
            ("nested rollback probe",),
        ).fetchone()[0]
        self.assertEqual(count, 0)
        self.assertFalse(database.connection.in_transaction)

    def test_outer_transaction_continues_after_inner_rollback(self):
        database = self.manager.database
        original = database.add_memory_fact
        failed_memory = self.memory(
            "failed nested write",
            [self.fact("failure probe", "like")],
        )
        valid_memory = self.memory("valid nested write", [])

        def fail_insert(*args, **kwargs):
            raise RuntimeError("injected nested insert failure")

        with database.atomic_write():
            database.add_memory_fact = fail_insert
            try:
                with self.assertRaisesRegex(RuntimeError, "injected nested insert failure"):
                    database.save_memory(failed_memory)
            finally:
                database.add_memory_fact = original

            self.assertTrue(database.connection.in_transaction)
            database.save_memory(valid_memory)

        contents = {
            row["content"]
            for row in database.cursor.execute(
                "SELECT content FROM memories"
            ).fetchall()
        }
        self.assertNotIn("failed nested write", contents)
        self.assertIn("valid nested write", contents)
        self.assertFalse(database.connection.in_transaction)

    def test_commit_failure_rolls_back_and_is_not_reported_as_success(self):
        database = self.manager.database
        proxy = self.inject_connection_failures()
        proxy.fail_commits = 1
        memory = self.memory("commit failure probe", [])

        with self.assertRaisesRegex(sqlite3.OperationalError, "injected commit failure"):
            database.save_memory(memory)

        count = database.cursor.execute(
            "SELECT COUNT(*) FROM memories WHERE content = ?",
            ("commit failure probe",),
        ).fetchone()[0]
        self.assertEqual(count, 0)
        self.assertFalse(database.connection.in_transaction)
        self.assertFalse(database._transaction_unusable)

        recovered = self.memory("write after recovered commit", [])
        database.save_memory(recovered)
        self.assertIsNotNone(database.get_memory(recovered.id))

    def test_commit_error_after_effect_marks_outcome_uncertain(self):
        database = self.manager.database
        proxy = self.inject_connection_failures()
        proxy.fail_after_commits = 1

        with self.assertRaisesRegex(RuntimeError, "uncertain outcome"):
            database.save_memory(self.memory("post-commit failure probe", []))

        self.assertFalse(database.connection.in_transaction)
        self.assertTrue(database._transaction_unusable)
        with self.assertRaisesRegex(RuntimeError, "state is uncertain"):
            database.save_memory(self.memory("must not write", []))

    def test_savepoint_release_failure_rolls_back_nested_write(self):
        database = self.manager.database
        proxy = self.inject_connection_failures()
        proxy.cursor_proxy.fail_releases = 1
        memory = self.memory("release failure probe", [])

        with self.assertRaisesRegex(sqlite3.OperationalError, "injected savepoint release failure"):
            with database.atomic_write():
                database.save_memory(memory)

        count = database.cursor.execute(
            "SELECT COUNT(*) FROM memories WHERE content = ?",
            ("release failure probe",),
        ).fetchone()[0]
        self.assertEqual(count, 0)
        self.assertFalse(database.connection.in_transaction)
        self.assertFalse(database._transaction_unusable)

    def test_uncertain_savepoint_rollback_disables_future_writes(self):
        database = self.manager.database
        proxy = self.inject_connection_failures()
        proxy.cursor_proxy.fail_releases = 1
        proxy.cursor_proxy.fail_rollback_to = 1
        memory = self.memory("uncertain savepoint probe", [])

        with self.assertRaisesRegex(RuntimeError, "connection is unusable"):
            with database.atomic_write():
                database.save_memory(memory)

        self.assertTrue(database._transaction_unusable)
        with self.assertRaisesRegex(RuntimeError, "state is uncertain"):
            database.save_memory(self.memory("must not write", []))

    def test_uncertain_connection_rollback_disables_future_writes(self):
        database = self.manager.database
        proxy = self.inject_connection_failures()
        proxy.fail_commits = 1
        proxy.fail_rollbacks = 1

        with self.assertRaisesRegex(RuntimeError, "connection is unusable"):
            database.save_memory(self.memory("uncertain commit probe", []))

        self.assertTrue(database._transaction_unusable)
        with self.assertRaisesRegex(RuntimeError, "state is uncertain"):
            database.save_memory(self.memory("must not write", []))

    def test_save_memory_rolls_back_fact_and_evidence_failure(self):
        database = self.manager.database
        original = database._write_fact_revision

        def fail_revision(*args, **kwargs):
            raise RuntimeError("injected revision failure")

        database._write_fact_revision = fail_revision
        memory = self.memory("evidence transaction probe", [self.fact("probe", "like")])
        try:
            with self.assertRaisesRegex(RuntimeError, "injected revision failure"):
                database.save_memory(memory, evidence=["fictional evidence"])
        finally:
            database._write_fact_revision = original

        for table in ("memories", "memory_facts", "fact_evidence"):
            count = database.cursor.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            self.assertEqual(count, 0, table)
        self.assertFalse(database.connection.in_transaction)

    def test_add_fact_evidence_rolls_back_when_revision_fails(self):
        existing = self.create_like()
        database = self.manager.database
        before = len(database.get_fact_evidence(existing.facts[0].id))
        original = database._write_fact_revision

        def fail_revision(*args, **kwargs):
            raise RuntimeError("injected evidence revision failure")

        database._write_fact_revision = fail_revision
        try:
            with self.assertRaisesRegex(RuntimeError, "injected evidence revision failure"):
                database.add_fact_evidence(existing.facts[0].id, "private sentinel evidence")
        finally:
            database._write_fact_revision = original

        self.assertEqual(
            len(database.get_fact_evidence(existing.facts[0].id)),
            before,
        )
        self.assertFalse(database.connection.in_transaction)

    def test_remember_rolls_back_reinforcement_if_new_fact_insert_fails(self):
        existing = self.create_like()
        database = self.manager.database
        fact_id = existing.facts[0].id
        original_history = database.get_fact_history(fact_id)
        original = database.add_memory_fact

        def fail_insert(*args, **kwargs):
            raise RuntimeError("injected remember insert failure")

        database.add_memory_fact = fail_insert
        memory = self.memory(
            "pizza and coffee preferences",
            [
                self.fact("pizza", "like"),
                self.fact("coffee", "like"),
            ],
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "injected remember insert failure"):
                self.manager.remember(memory)
        finally:
            database.add_memory_fact = original

        self.assertEqual(len(database.find_memories()), 1)
        self.assertEqual(database.get_fact_evidence(fact_id), [])
        self.assertEqual(database.get_fact_history(fact_id), original_history)
        self.assertEqual(database.get_fact(fact_id).confidence, 0.7)
        self.assertFalse(database.connection.in_transaction)

    def test_conflict_resolution_rolls_back_if_graph_write_fails(self):
        database = self.manager.database
        existing = Memory(
            content="existing preference",
            memory_type="preference",
            importance=0.1,
            emotion="neutral",
            emotional_intensity=0.0,
            facts=[
                MemoryFact(
                    target="pizza",
                    relation="like",
                    temporal_context="current",
                    confidence=0.1,
                    importance=0.1,
                    fact_type="preference",
                )
            ],
        )
        database.save_memory(existing)
        incoming = Memory(
            content="new preference",
            memory_type="preference",
            importance=0.99,
            emotion="neutral",
            emotional_intensity=0.0,
            facts=[
                MemoryFact(
                    target="pizza",
                    relation="dislike",
                    temporal_context="current",
                    confidence=0.99,
                    importance=0.99,
                    fact_type="preference",
                )
            ],
        )
        database.save_memory(incoming)
        incoming_fact = database.get_memory(incoming.id).facts[0]
        original = database._upsert_graph_edge

        def fail_graph(*args, **kwargs):
            raise RuntimeError("injected graph write failure")

        database._upsert_graph_edge = fail_graph
        try:
            with self.assertRaisesRegex(RuntimeError, "injected graph write failure"):
                self.manager.resolve_conflicts_for_fact(incoming_fact)
        finally:
            database._upsert_graph_edge = original

        self.assertEqual(database.get_conflicts(), [])
        statuses = {
            fact.id: fact.status
            for fact in database.find_facts_by_target("pizza")
        }
        self.assertEqual(statuses, {existing.facts[0].id: "active", incoming_fact.id: "active"})
        self.assertFalse(database.connection.in_transaction)

    def test_apply_decay_rolls_back_when_second_archive_fails(self):
        database = self.manager.database
        memory = self.memory(
            "two low importance facts",
            [
                self.fact("pizza", "like", importance=0.1),
                self.fact("coffee", "like", importance=0.1),
            ],
        )
        database.save_memory(memory)
        original = database.update_memory_fact
        call_count = 0

        def fail_second(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("injected second archive failure")
            return original(*args, **kwargs)

        database.update_memory_fact = fail_second
        try:
            with self.assertRaisesRegex(RuntimeError, "injected second archive failure"):
                self.manager.apply_decay(
                    now=datetime(2035, 1, 1, tzinfo=timezone.utc),
                    archive_below=1.0,
                )
        finally:
            database.update_memory_fact = original

        statuses = {
            fact.target: fact.status
            for fact in database.get_memory(memory.id).facts
        }
        self.assertEqual(statuses, {"pizza": "active", "coffee": "active"})
        self.assertFalse(database.connection.in_transaction)

    def test_save_memory_rejects_invalid_values_before_writing(self):
        database = self.manager.database
        invalid_memories = (
            self.memory("   ", [self.fact("pizza", "like")]),
            self.memory("invalid metadata", [self.fact("pizza", "like")]),
        )
        invalid_memories[1].metadata = {"not_json": object()}

        for memory in invalid_memories:
            with self.assertRaises(ValueError):
                database.save_memory(memory)

        count = database.cursor.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]
        self.assertEqual(count, 0)
        self.assertFalse(database.connection.in_transaction)

    def test_last_accessed_at_accepts_none_and_existing_text(self):
        database = self.manager.database
        missing = self.memory("no access timestamp", [])
        preserved = self.memory("legacy timestamp", [])
        preserved.last_accessed_at = "legacy date text"

        database.save_memory(missing)
        database.save_memory(preserved)

        self.assertIsNone(database.get_memory(missing.id).last_accessed_at)
        self.assertEqual(
            database.get_memory(preserved.id).last_accessed_at,
            "legacy date text",
        )

    def test_last_accessed_at_rejects_wrong_type_before_writing(self):
        database = self.manager.database
        memory = self.memory("invalid access timestamp", [])
        memory.last_accessed_at = 123

        with self.assertRaisesRegex(ValueError, "memory.last_accessed_at must be text"):
            database.save_memory(memory)

        count = database.cursor.execute(
            "SELECT COUNT(*) FROM memories WHERE content = ?",
            ("invalid access timestamp",),
        ).fetchone()[0]
        self.assertEqual(count, 0)
        self.assertFalse(database.connection.in_transaction)

    def test_last_accessed_at_respects_sqlite_text_limit_before_writing(self):
        database = self.manager.database
        memory = self.memory("short content", [])
        memory.last_accessed_at = "x" * 80
        database._sqlite_length_limit = lambda: 64

        with self.assertRaisesRegex(ValueError, "SQLite length limit"):
            database.save_memory(memory)

        count = database.cursor.execute(
            "SELECT COUNT(*) FROM memories WHERE content = ?",
            ("short content",),
        ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_strict_json_preserves_valid_nested_metadata(self):
        database = self.manager.database
        memory = self.memory("valid nested metadata", [])
        memory.metadata = {"items": [1, "two", {"enabled": True}]}

        database.save_memory(memory)

        self.assertEqual(
            database.get_memory(memory.id).metadata,
            {"items": [1, "two", {"enabled": True}]},
        )

    def test_strict_json_rejects_non_finite_values_at_any_depth(self):
        database = self.manager.database
        non_finite_values = (float("nan"), float("inf"), float("-inf"))

        for index, value in enumerate(non_finite_values):
            with self.subTest(value=repr(value)):
                memory = self.memory(f"invalid nested metadata {index}", [])
                memory.metadata = {"outer": [{"inner": [value]}]}
                with self.assertRaisesRegex(ValueError, "JSON serializable"):
                    database.save_memory(memory)

        count = database.cursor.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]
        self.assertEqual(count, 0)
        self.assertFalse(database.connection.in_transaction)

    def test_strict_json_metadata_failure_rolls_back_memory_and_facts(self):
        database = self.manager.database
        memory = self.memory(
            "invalid fact metadata",
            [self.fact("metadata probe", "like")],
        )
        memory.facts[0].metadata = {"nested": [{"bad": float("inf")}]}

        with self.assertRaisesRegex(ValueError, "JSON serializable"):
            database.save_memory(memory)

        for table in ("memories", "memory_facts", "fact_evidence"):
            count = database.cursor.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            self.assertEqual(count, 0, table)

    def test_direct_fact_write_preserves_none_defaults(self):
        database = self.manager.database
        memory = self.memory("defaults", [])
        memory_id = database.save_memory(memory)

        fact_id = database.add_memory_fact(
            memory_id=memory_id,
            target="default probe",
            relation="like",
            emotion=None,
            temporal_context=None,
            subject=None,
            status=None,
            source=None,
        )

        fact = database.get_fact(fact_id)
        self.assertEqual(fact.emotion, "neutral")
        self.assertEqual(fact.temporal_context, "unknown")
        self.assertEqual(fact.subject, "user")
        self.assertEqual(fact.status, "active")
        self.assertEqual(fact.source, "user_statement")

    def test_memory_and_fact_reject_unsupported_status_before_writing(self):
        database = self.manager.database
        invalid_memory = self.memory("invalid memory status", [])
        invalid_memory.status = "open"
        with self.assertRaisesRegex(ValueError, "memory.status"):
            database.save_memory(invalid_memory)

        memory_id = database.save_memory(self.memory("valid parent", []))
        with self.assertRaisesRegex(ValueError, "fact.status"):
            database.add_memory_fact(
                memory_id=memory_id,
                target="probe",
                relation="like",
                status="open",
            )

        count = database.cursor.execute(
            "SELECT COUNT(*) FROM memory_facts"
        ).fetchone()[0]
        self.assertEqual(count, 0)
        self.assertFalse(database.connection.in_transaction)

    def test_save_memory_uses_sqlite_configured_text_limit(self):
        database = self.manager.database
        memory = self.memory("x" * 32, [self.fact("probe", "like")])
        database._sqlite_length_limit = lambda: 16

        with self.assertRaisesRegex(ValueError, "SQLite length limit"):
            database.save_memory(memory)

        count = database.cursor.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_batch_update_validates_all_facts_before_writing(self):
        existing = self.create_like()
        original_history = self.manager.database.get_fact_history(existing.facts[0].id)
        update = self.memory(
            "multi-fact update",
            [
                self.fact("pizza", "dislike", negation=True),
                self.fact("coffee", "like", confidence=float("inf")),
            ],
        )

        with self.assertRaisesRegex(ValueError, "must be finite"):
            self.manager.update_memory_fact(existing, update)

        persisted = self.manager.database.get_memory(existing.id)
        self.assertEqual(persisted.facts[0].relation, "like")
        self.assertEqual(
            self.manager.database.get_fact_history(existing.facts[0].id),
            original_history,
        )

    def test_batch_update_rolls_back_when_second_fact_fails(self):
        existing = self.memory(
            "original preferences",
            [self.fact("pizza", "like"), self.fact("coffee", "like")],
        )
        self.apply("create", existing)
        update = self.memory(
            "updated preferences",
            [
                self.fact("pizza", "dislike", negation=True),
                self.fact("coffee", "dislike", negation=True),
            ],
        )
        database = self.manager.database
        original = database.update_memory_fact
        call_count = 0

        def fail_second(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("injected second fact failure")
            return original(*args, **kwargs)

        database.update_memory_fact = fail_second
        try:
            with self.assertRaisesRegex(RuntimeError, "injected second fact failure"):
                self.manager.update_memory_fact(existing, update)
        finally:
            database.update_memory_fact = original

        persisted = database.get_memory(existing.id)
        self.assertEqual(
            {fact.target: fact.relation for fact in persisted.facts},
            {"pizza": "like", "coffee": "like"},
        )
        self.assertFalse(database.connection.in_transaction)

    def test_multiple_facts_update_in_one_memory(self):
        existing = self.memory(
            "preferences",
            [self.fact("pizza", "like"), self.fact("coffee", "like")],
        )
        self.apply("create", existing)
        update = self.memory(
            "changed preferences",
            [
                self.fact("pizza", "dislike", negation=True),
                self.fact("coffee", "dislike", negation=True),
            ],
        )
        result = self.apply("update", update, self.manager.database.get_memory(existing.id))
        self.assertEqual(result["memory"].id, existing.id)
        self.assertEqual({fact.relation for fact in result["memory"].facts}, {"dislike"})
        self.assertEqual(len(self.manager.recall()), 1)

    def test_parser_cases(self):
        context_analyzer = ContextAnalyzer()
        emotion_analyzer = EmotionAnalyzer()
        interpreter = MemoryInterpreter()
        cases = (
            ("agora eu gosto de pizza", "like", False, 0.7),
            ("agora eu não gosto de pizza", "dislike", True, 0.7),
            ("agora eu nao gosto de pizza", "dislike", True, 0.7),
            ("eu acho que nao gosto mais de pizza na real", "dislike", True, 0.65),
            ("eu acho que gosto de pizza", "like", False, 0.65),
        )
        for text, relation, negation, confidence in cases:
            context = context_analyzer.analyze(text)
            emotions = []
            for part in context["parts"]:
                emotion = emotion_analyzer.analyze(part["text"], part)
                emotions.append({
                    "text": part["text"],
                    "emotion": emotion["emotion"],
                    "emotional_intensity": emotion["emotional_intensity"],
                    "context": part,
                })
            facts = interpreter.interpret(text, context, emotions)["facts"]
            self.assertEqual(len(facts), 1)
            self.assertEqual(facts[0].target, "pizza")
            self.assertEqual(facts[0].relation, relation)
            self.assertEqual(facts[0].negation, negation)
            self.assertEqual(facts[0].confidence, confidence)

    def test_memory_logs_omit_text_context_and_targets(self):
        text_sentinel = "PRIVATE_TEXT_SENTINEL"
        context_sentinel = "PRIVATE_CONTEXT_SENTINEL"
        target_sentinel = "PRIVATE_TARGET_SENTINEL"

        with self.assertLogs("davios.memory.emotion", level="DEBUG") as captured:
            EmotionAnalyzer().analyze(
                text_sentinel,
                {"private_context": context_sentinel},
            )
        emotion_logs = "\n".join(captured.output)
        self.assertIn("input_chars=", emotion_logs)
        self.assertNotIn(text_sentinel, emotion_logs)
        self.assertNotIn(context_sentinel, emotion_logs)

        fact = self.fact(target_sentinel, "like")
        with self.assertLogs("davios.memory.manager", level="DEBUG") as captured:
            self.manager._flow_log("probe", None, fact, fact, "none")
        memory_logs = "\n".join(captured.output)
        self.assertIn("operation=probe", memory_logs)
        self.assertNotIn(target_sentinel, memory_logs)

    def test_recall_candidates_match_legacy_ranking_on_large_mixed_data(self):
        database = self.manager.database
        targets = (
            "pizza",
            "café",
            "alpha beta",
            "pizza com queijo",
            "projeto local",
            "sem relação",
        )
        fact_statuses = (
            "active",
            "conflicted",
            "superseded",
            "archived",
            "inactive",
            "retracted",
        )
        base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for index in range(96):
            target = targets[index % len(targets)]
            fact = self.fact(
                target,
                "like" if index % 3 else "working_on",
                temporal_context=("current", "past", "future", "unknown")[index % 4],
                confidence=(index % 10) / 10,
                importance=(index % 8) / 8,
                status=fact_statuses[index % len(fact_statuses)],
            )
            memory = self.memory(f"entry {index} mentions {target}", [fact])
            memory.status = "archived" if index % 19 == 0 else "active"
            memory.base_importance = (index % 10) / 10
            evidence = [f"fictional evidence {index}"] if index % 7 == 0 else None
            database.save_memory(
                memory,
                evidence=evidence,
                now=base_time + timedelta(days=index),
                allow_duplicate_content=True,
            )

        fixed_now = datetime(2026, 10, 1, tzinfo=timezone.utc)

        def legacy_result(query, limit, include_inactive):
            query_text = self.manager.normalize_fact_target(query)
            query_targets = {query_text} if query_text else set()
            query_relations = set()
            query_tokens = set(query_text.split())
            ranked = []
            for memory in database.find_memories(include_archived=True):
                if memory.status == "archived" and not include_inactive:
                    continue
                for fact in memory.facts:
                    if not include_inactive and fact.status not in {"active", "conflicted"}:
                        continue
                    target = self.manager.normalize_fact_target(fact.target)
                    target_tokens = set(target.split())
                    reasons = []
                    score = 0.0
                    if target in query_targets:
                        score += 0.48
                        reasons.append("alvo exato")
                    elif target and target in query_text:
                        score += 0.38
                        reasons.append("alvo mencionado")
                    elif query_tokens and target_tokens:
                        overlap = len(query_tokens & target_tokens) / len(target_tokens)
                        if overlap:
                            score += 0.28 * overlap
                            reasons.append("sobreposição lexical")
                    if (fact.relation or "").casefold() in query_relations:
                        score += 0.10
                        reasons.append("relação compatível")
                    if query_text and query_text in memory.content.casefold():
                        score += 0.08
                        reasons.append("texto da memória")
                    temporal_kind = self.manager._temporal_kind(fact.temporal_context)
                    if temporal_kind == "current":
                        score += 0.05
                        reasons.append("contexto atual")
                    elif temporal_kind == "past":
                        score += 0.01
                    effective = self.manager.effective_importance(fact, now=fixed_now)
                    score += effective * 0.35
                    reasons.append("importância dinâmica")
                    if fact.status == "conflicted":
                        score -= 0.12
                        reasons.append("penalidade de conflito aberto")
                    ranked.append(
                        {
                            "memory": memory,
                            "fact": fact,
                            "score": round(max(0.0, score), 6),
                            "effective_importance": effective,
                            "reasons": reasons,
                        }
                    )
            ranked.sort(
                key=lambda item: (
                    -item["score"],
                    item["fact"].created_at or "",
                    item["fact"].id or 0,
                ),
                reverse=False,
            )
            return ranked[: max(0, limit)]

        for query, include_inactive in (
            ("pizza", False),
            ("café projeto", False),
            ("nada correspondente", False),
            ("pizza", True),
        ):
            expected = legacy_result(query, 17, include_inactive)
            actual = self.manager.recall_relevant(
                query,
                limit=17,
                include_inactive=include_inactive,
                now=fixed_now,
            )
            expected_signature = [
                (
                    item["memory"].id,
                    item["fact"].id,
                    item["score"],
                    item["effective_importance"],
                    item["reasons"],
                )
                for item in expected
            ]
            actual_signature = [
                (
                    item["memory"].id,
                    item["fact"].id,
                    item["score"],
                    item["effective_importance"],
                    item["reasons"],
                )
                for item in actual
            ]
            self.assertEqual(actual_signature, expected_signature)

    def test_g4f1_recall_touch_increments_once_per_selected_memory(self):
        database = self.manager.database
        first = self.memory(
            "g4f1 first mentions pizza",
            [self.fact("pizza", "like"), self.fact("pizza", "like")],
        )
        second = self.memory("g4f1 second mentions pizza", [self.fact("pizza", "like")])
        first_id = database.save_memory(first, allow_duplicate_content=True)
        second_id = database.save_memory(second, allow_duplicate_content=True)
        untouched = self.memory("g4f1 untouched", [self.fact("cha verde", "like")])
        untouched_id = database.save_memory(untouched, allow_duplicate_content=True)

        before = {
            memory_id: database.get_memory(memory_id).access_count
            for memory_id in (first_id, second_id, untouched_id)
        }

        results = self.manager.recall_relevant("pizza", limit=10)
        selected_ids = {item["memory"].id for item in results}
        self.assertIn(first_id, selected_ids)
        self.assertIn(second_id, selected_ids)

        after = {
            memory_id: database.get_memory(memory_id)
            for memory_id in (first_id, second_id, untouched_id)
        }
        for memory_id in selected_ids:
            self.assertEqual(
                after[memory_id].access_count,
                before[memory_id] + 1,
            )
            self.assertIsNotNone(after[memory_id].last_accessed_at)
        for memory_id in (first_id, second_id, untouched_id):
            if memory_id not in selected_ids:
                self.assertEqual(
                    after[memory_id].access_count,
                    before[memory_id],
                )
        self.assertFalse(database.connection.in_transaction)

    def test_g4f1_recall_touch_sets_last_accessed_at_to_recall_instant(self):
        database = self.manager.database
        memory = self.memory("g4f1 instant mentions pizza", [self.fact("pizza", "like")])
        memory_id = database.save_memory(memory, allow_duplicate_content=True)
        before_count = database.get_memory(memory_id).access_count
        instant = datetime(2026, 5, 4, 3, 2, 1, tzinfo=timezone.utc)

        results = self.manager.recall_relevant("pizza", limit=10, now=instant)
        self.assertTrue(any(item["memory"].id == memory_id for item in results))

        refreshed = database.get_memory(memory_id)
        self.assertEqual(refreshed.access_count, before_count + 1)
        self.assertEqual(refreshed.last_accessed_at, database._now(instant))
        self.assertFalse(database.connection.in_transaction)

    def test_g4f1_recall_does_not_change_updated_at_content_or_semantics(self):
        database = self.manager.database
        memory = self.memory("g4f1 stable mentions pizza", [self.fact("pizza", "like")])
        memory_id = database.save_memory(memory, allow_duplicate_content=True)
        snapshot = database.get_memory(memory_id)
        snapshot_fact = snapshot.facts[0].snapshot()

        results = self.manager.recall_relevant("pizza", limit=10)
        self.assertTrue(any(item["memory"].id == memory_id for item in results))

        refreshed = database.get_memory(memory_id)
        self.assertEqual(refreshed.updated_at, snapshot.updated_at)
        self.assertEqual(refreshed.content, snapshot.content)
        self.assertEqual(refreshed.status, snapshot.status)
        self.assertEqual(len(refreshed.facts), 1)
        self.assertEqual(refreshed.facts[0].snapshot(), snapshot_fact)
        self.assertEqual(refreshed.access_count, snapshot.access_count + 1)
        self.assertFalse(database.connection.in_transaction)

    def test_g4f1_active_preferences_filter_by_subject_and_family(self):
        database = self.manager.database
        user_pizza = MemoryFact(target="pizza", relation="like", subject="user")
        ana_pizza = MemoryFact(target="pizza", relation="like", subject="ana")
        user_work = MemoryFact(target="netoptimizer", relation="working_on", subject="user")
        database.save_memory(
            self.memory("g4f1 prefs", [user_pizza, ana_pizza, user_work]),
            allow_duplicate_content=True,
        )

        user_preferences = database.find_active_preferences("user", "preference")
        self.assertEqual(
            [(fact.subject, fact.target) for fact in user_preferences],
            [("user", "pizza")],
        )
        ana_preferences = database.find_active_preferences("ana", "preference")
        self.assertEqual(
            [(fact.subject, fact.target) for fact in ana_preferences],
            [("ana", "pizza")],
        )
        working = database.find_active_preferences("user", "working_on")
        self.assertEqual([fact.target for fact in working], ["netoptimizer"])
        self.assertFalse(database.connection.in_transaction)

    def test_g4f1_active_preferences_isolate_identity_working_on_studies(self):
        database = self.manager.database
        database.save_memory(
            self.memory(
                "g4f1 families",
                [
                    MemoryFact(target="davi", relation="identity", subject="user"),
                    MemoryFact(target="netoptimizer", relation="working_on", subject="user"),
                    MemoryFact(target="ingles", relation="studies", subject="user"),
                ],
            ),
            allow_duplicate_content=True,
        )

        identity = database.find_active_preferences("user", "identity")
        working = database.find_active_preferences("user", "working_on")
        studies = database.find_active_preferences("user", "studies")
        preferences = database.find_active_preferences("user", "preference")
        self.assertEqual([fact.target for fact in identity], ["davi"])
        self.assertEqual([fact.target for fact in working], ["netoptimizer"])
        self.assertEqual([fact.target for fact in studies], ["ingles"])
        self.assertEqual(preferences, [])
        self.assertFalse(database.connection.in_transaction)

    def test_g4f1_active_preferences_match_subject_and_relation_case_insensitively(self):
        database = self.manager.database
        database.save_memory(
            self.memory(
                "g4f1 case",
                [
                    MemoryFact(target="Pizza", relation="Like", subject="User"),
                    MemoryFact(target="netoptimizer", relation="WORKING_ON", subject="USER"),
                ],
            ),
            allow_duplicate_content=True,
        )

        preferences = database.find_active_preferences("uSeR", "preference")
        self.assertEqual(
            [(fact.subject, fact.target) for fact in preferences],
            [("User", "Pizza")],
        )
        working = database.find_active_preferences("user", "working_on")
        self.assertEqual([fact.target for fact in working], ["netoptimizer"])
        self.assertFalse(database.connection.in_transaction)

    def test_g4f1_active_preferences_empty_and_multiple_in_id_order(self):
        database = self.manager.database
        self.assertEqual(database.find_active_preferences("user", "preference"), [])

        database.save_memory(
            self.memory(
                "g4f1 ordered",
                [
                    self.fact("cha", "like"),
                    self.fact("cafe", "like"),
                    self.fact("pizza", "like"),
                ],
            ),
            allow_duplicate_content=True,
        )
        preferences = database.find_active_preferences("user", "preference")
        self.assertEqual([fact.target for fact in preferences], ["cha", "cafe", "pizza"])
        self.assertEqual(
            [fact.id for fact in preferences],
            sorted(fact.id for fact in preferences),
        )
        self.assertFalse(database.connection.in_transaction)

    def test_g4f1_recall_objects_match_hydrated_database_records(self):
        database = self.manager.database
        memory = self.memory(
            "g4f1 hydrated mentions pizza",
            [self.fact("pizza", "like"), self.fact("cafe", "like")],
        )
        memory_id = database.save_memory(memory, allow_duplicate_content=True)

        results = self.manager.recall_relevant("pizza", limit=10)
        matches = [item for item in results if item["memory"].id == memory_id]
        self.assertTrue(matches)
        for item in matches:
            hydrated_memory = database.get_memory(item["memory"].id)
            hydrated_fact = database.get_fact(item["fact"].id)
            self.assertEqual(item["memory"].id, hydrated_memory.id)
            self.assertEqual(item["memory"].content, hydrated_memory.content)
            self.assertEqual(item["memory"].status, hydrated_memory.status)
            self.assertEqual(item["fact"].id, hydrated_fact.id)
            self.assertEqual(item["fact"].target, hydrated_fact.target)
            self.assertEqual(item["fact"].relation, hydrated_fact.relation)
            self.assertEqual(item["fact"].status, hydrated_fact.status)
            self.assertIn(item["fact"].id, {fact.id for fact in hydrated_memory.facts})
        self.assertFalse(database.connection.in_transaction)

    def test_real_pipeline_updates_one_memory(self):
        context_analyzer = ContextAnalyzer()
        emotion_analyzer = EmotionAnalyzer()
        interpreter = MemoryInterpreter()

        def build_memory(text):
            context = context_analyzer.analyze(text)
            emotions = []
            for part in context["parts"]:
                emotion = emotion_analyzer.analyze(part["text"], part)
                emotions.append({
                    "text": part["text"],
                    "emotion": emotion["emotion"],
                    "emotional_intensity": emotion["emotional_intensity"],
                    "context": part,
                })
            analysis = interpreter.interpret(text, context, emotions)
            return context, self.memory(text, analysis["facts"])

        first_context, first_memory = build_memory("agora eu gosto de pizza")
        self.reasoning.memory_operation = "create"
        self.manager.apply_memory_operation(self.reasoning, first_memory)

        second_context, second_memory = build_memory(
            "eu acho que nao gosto mais de pizza na real"
        )
        existing = self.manager.find_memory_for_facts(second_memory.facts)
        self.assertIsNotNone(existing)
        self.reasoning.memory_operation = "update"
        result = self.manager.apply_memory_operation(
            self.reasoning,
            second_memory,
            existing,
        )
        self.assertEqual(result["action"], "update")
        self.assertEqual(result["memory"].id, 1)
        self.assertEqual(result["memory"].facts[0].target, "pizza")
        self.assertEqual(result["memory"].facts[0].relation, "dislike")
        self.assertTrue(result["memory"].facts[0].negation)
        self.assertEqual(len(self.manager.recall()), 1)

    def test_match_and_reasoning_propagate_ids_and_reinforce(self):
        existing = self.create_like()
        new_memory = self.memory("eu gosto de pizza", [self.fact("pizza", "like")])
        match = self.manager.match_memory_for_facts(new_memory.facts)
        self.assertEqual(match["existing_memory"].id, existing.id)
        self.assertEqual(match["matching_fact"].id, existing.facts[0].id)

        decision = ReasoningEngine().evaluate_input(
            new_memory.content,
            relevant_memories=[match["matching_fact"]],
            current_facts=new_memory.facts,
            existing_memory=match["existing_memory"],
            matching_fact=match["matching_fact"],
        )
        self.assertEqual(decision.memory_operation, "reinforce")
        self.assertEqual(decision.existing_memory_id, existing.id)
        self.assertEqual(decision.matching_fact_id, existing.facts[0].id)
        result = self.manager.apply_memory_operation(
            decision,
            new_memory,
            match["existing_memory"],
        )
        self.assertEqual(result["action"], "reinforce")
        self.assertEqual(len(self.manager.recall()), 1)

    def test_reasoning_updates_relation_change_and_never_creates(self):
        existing = self.create_like()
        new_memory = self.memory(
            "eu nao gosto mais de pizza",
            [self.fact("pizza", "dislike", negation=True)],
        )
        match = self.manager.match_memory_for_facts(new_memory.facts)
        decision = ReasoningEngine().evaluate_input(
            new_memory.content,
            relevant_memories=[match["matching_fact"]],
            current_facts=new_memory.facts,
            existing_memory=match["existing_memory"],
            matching_fact=match["matching_fact"],
        )
        self.assertEqual(decision.memory_operation, "update")
        self.assertEqual(decision.existing_memory_id, existing.id)
        result = self.manager.apply_memory_operation(
            decision,
            new_memory,
            match["existing_memory"],
        )
        self.assertEqual(result["memory"].id, existing.id)
        self.assertEqual(len(self.manager.recall()), 1)

    def test_required_preference_evolution_sequence(self):
        context_analyzer = ContextAnalyzer()
        emotion_analyzer = EmotionAnalyzer()
        interpreter = MemoryInterpreter()

        def build(text):
            context = context_analyzer.analyze(text)
            emotions = []
            for part in context["parts"]:
                emotion = emotion_analyzer.analyze(part["text"], part)
                emotions.append({
                    "text": part["text"],
                    "emotion": emotion["emotion"],
                    "emotional_intensity": emotion["emotional_intensity"],
                    "context": part,
                })
            facts = interpreter.interpret(text, context, emotions)["facts"]
            return self.memory(text, facts)

        first = build("eu gosto de pizza")
        self.apply("create", first)
        for text, operation, relation, negation in (
            ("eu nao gosto mais de pizza", "update", "dislike", True),
            ("eu realmente nao gosto de pizza", "reinforce", "dislike", True),
            ("na verdade voltei a gostar de pizza", "update", "like", False),
        ):
            new_memory = build(text)
            self.assertEqual(len(new_memory.facts), 1)
            self.assertEqual(new_memory.facts[0].target, "pizza")
            existing = self.manager.find_memory_for_facts(new_memory.facts)
            result = self.apply(operation, new_memory, existing)
            self.assertEqual(result["action"], operation)
            self.assertEqual(result["memory"].facts[0].relation, relation)
            self.assertEqual(result["memory"].facts[0].negation, negation)
        self.assertEqual(len(self.manager.recall()), 1)


if __name__ == "__main__":
    unittest.main()
