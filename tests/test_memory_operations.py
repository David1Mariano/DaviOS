import os
import tempfile
import unittest

from core.reasoning import ReasoningResult
from core.reasoning import ReasoningEngine
from memory.context_analyzer import ContextAnalyzer
from memory.emotion_analyzer import EmotionAnalyzer
from memory.memory import Memory
from memory.memory_fact import MemoryFact
from memory.memory_interpreter import MemoryInterpreter
from memory.memory_manager import MemoryManager


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
