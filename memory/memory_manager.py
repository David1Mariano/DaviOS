"""Políticas de domínio para gravar, recuperar e resolver memórias."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from memory.context_analyzer import ContextAnalyzer
from memory.database import Database
from memory.emotion_analyzer import EmotionAnalyzer
from memory.memory import Memory
from memory.memory_fact import MemoryFact


class MemoryManager:
    """Fachada compatível para o armazenamento de memória.

    A classe concentra a política que não deve ficar no SQLite: evolução de
    preferências, resolução de conflitos, decay e ranking por relevância.
    """

    POSITIVE_PREFERENCES = {"like", "love", "prefer", "favorite"}
    NEGATIVE_PREFERENCES = {"dislike", "hate", "avoid"}
    TEMPORAL_WORDS = {
        "agora",
        "atualmente",
        "hoje",
        "antes",
        "antigamente",
        "ontem",
        "amanhã",
        "amanha",
        "sempre",
        "nunca",
    }
    HALF_LIFE_DAYS = {
        "preference": 365.0,
        "episodic": 30.0,
        "fact": 180.0,
        "factual": 180.0,
        "mixed": 120.0,
    }

    def __init__(
        self,
        db_path: str = "memory.db",
        *,
        database: Optional[Database] = None,
        now_provider: Optional[Callable[[], datetime]] = None,
    ):
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self.database = database or Database(
            db_path=db_path,
            now_provider=self._now_provider,
        )
        self.emotion_analyzer = EmotionAnalyzer()
        self.context_analyzer = ContextAnalyzer()

    # ------------------------------------------------------------------
    # Backwards-compatible discovery helpers
    # ------------------------------------------------------------------

    def recall(self, target: Optional[str] = None):
        """Retorna todos os fatos de um alvo ou todas as memórias.

        ``recall_relevant`` é a API recomendada quando houver uma consulta a
        ranquear. Esta função mantém o formato simples usado pelo código legado.
        """

        if target is not None:
            return self.database.find_facts_by_target(target)
        return self.database.find_memories()

    def find_memories_for_facts(
        self,
        facts: Iterable[MemoryFact],
        *,
        limit: int = 10,
    ) -> list[Memory]:
        targets = [self.normalize_fact_target(fact.target) for fact in facts]
        targets = [target for target in targets if target]
        if not targets:
            return []
        ranked = self.recall_relevant(" ".join(targets), limit=limit)
        memories: list[Memory] = []
        seen: set[int] = set()
        for item in ranked:
            memory = item["memory"]
            if memory.id not in seen:
                seen.add(memory.id)
                memories.append(memory)
        return memories

    def find_memory_for_facts(self, facts: Iterable[MemoryFact]) -> Optional[Memory]:
        """Wrapper legado: devolve a melhor memória, não a primeira do banco."""

        memories = self.find_memories_for_facts(facts, limit=1)
        return memories[0] if memories else None

    def normalize_memory_facts(self, memory: Memory) -> Memory:
        """Normaliza grafia sem descartar fatos que compartilham um alvo."""

        normalized_facts: list[MemoryFact] = []
        for fact in memory.facts:
            target = self.normalize_fact_target(fact.target)
            if not target:
                continue
            fact.target = target
            fact.subject = self.normalize_subject(fact.subject)
            normalized_facts.append(fact)
        memory.facts = normalized_facts
        return memory

    def normalize_fact_target(self, target: Optional[str]) -> str:
        if not target:
            return ""
        words = re.findall(r"[^\W_]+", target.casefold(), flags=re.UNICODE)
        return " ".join(word for word in words if word not in self.TEMPORAL_WORDS).strip()

    @staticmethod
    def normalize_subject(subject: Optional[str]) -> str:
        return " ".join((subject or "user").casefold().split()) or "user"

    def facts_are_equal(self, existing_fact: MemoryFact, new_fact: MemoryFact) -> bool:
        """Compara duas observações completas, preservando tempo e negação."""

        return (
            self.normalize_fact_target(existing_fact.target)
            == self.normalize_fact_target(new_fact.target)
            and self.normalize_subject(existing_fact.subject)
            == self.normalize_subject(new_fact.subject)
            and (existing_fact.relation or "").casefold()
            == (new_fact.relation or "").casefold()
            and (existing_fact.value or "") == (new_fact.value or "")
            and (existing_fact.emotion or "") == (new_fact.emotion or "")
            and float(existing_fact.emotional_intensity or 0)
            == float(new_fact.emotional_intensity or 0)
            and (existing_fact.temporal_context or "unknown").casefold()
            == (new_fact.temporal_context or "unknown").casefold()
            and bool(existing_fact.negation) == bool(new_fact.negation)
        )

    def find_matching_fact(
        self,
        memory: Memory,
        new_fact: MemoryFact,
    ) -> Optional[MemoryFact]:
        """Encontra somente uma confirmação exata, nunca apenas o mesmo alvo."""

        for existing_fact in memory.facts:
            if self.facts_are_equal(existing_fact, new_fact):
                return existing_fact
        return None

    # ------------------------------------------------------------------
    # Storage and evidence reinforcement
    # ------------------------------------------------------------------

    def remember(
        self,
        memory: Memory,
        *,
        evidence: Optional[Iterable[dict[str, Any] | str]] = None,
        now: Optional[datetime | str] = None,
        allow_duplicate_content: bool = True,
    ) -> dict[str, Any]:
        """Registra uma observação, mantendo evolução e origem auditáveis.

        Declarações já conhecidas viram evidência/confirmacão. Declarações novas
        são adicionadas como fatos novos, inclusive quando o alvo já existe em
        outra memória.
        """

        self.normalize_memory_facts(memory)
        if not memory.facts:
            return {"operation": "ignored", "memory": None, "facts": []}

        timestamp = self._timestamp(now)
        new_facts: list[MemoryFact] = []
        reinforced: list[dict[str, Any]] = []
        default_evidence = list(evidence or [])
        if not default_evidence:
            default_evidence = [
                {
                    "content": memory.content,
                    "source": "user_statement",
                    "observed_at": timestamp,
                    "confidence": memory.confidence,
                }
            ]

        for fact in memory.facts:
            fact.fact_type = fact.fact_type or memory.memory_type
            candidates = self.database.find_facts_for_semantic_key(
                fact.subject,
                fact.target,
                fact.relation_family,
                include_inactive=False,
            )
            equivalent = next(
                (candidate for candidate in candidates if self.facts_are_equal(candidate, fact)),
                None,
            )
            if equivalent is None:
                new_facts.append(fact)
                continue
            used_evidence = fact.evidence or default_evidence
            for item in used_evidence:
                self.database.add_fact_evidence(equivalent.id, item, now=timestamp)
            evidence_count = len(self.database.get_fact_evidence(equivalent.id))
            reinforced_confidence = min(
                0.99,
                max(equivalent.confidence, fact.confidence) + min(0.12, evidence_count * 0.02),
            )
            self.database.update_memory_fact(
                memory_id=equivalent.memory_id,
                fact_id=equivalent.id,
                confidence=reinforced_confidence,
                reason="Declaração equivalente reforçada por nova evidência.",
                revision_type="confirmed",
                now=timestamp,
            )
            reinforced.append(
                {
                    "action": "reinforce",
                    "fact_id": equivalent.id,
                    "target": equivalent.target,
                    "relation": equivalent.relation,
                }
            )

        if not new_facts:
            return {
                "operation": "reinforced",
                "memory": None,
                "facts": reinforced,
            }

        record = Memory(
            content=memory.content,
            memory_type=memory.memory_type,
            importance=memory.importance,
            base_importance=memory.base_importance,
            emotion=memory.emotion,
            emotional_intensity=memory.emotional_intensity,
            confidence=memory.confidence,
            status=memory.status,
            metadata=memory.metadata,
            facts=new_facts,
        )
        memory_id = self.database.save_memory(
            record,
            evidence=default_evidence,
            now=timestamp,
            allow_duplicate_content=allow_duplicate_content,
        )
        if memory_id is None:
            return {
                "operation": "ignored",
                "memory": self.database.find_memory(memory),
                "facts": reinforced,
            }

        conflict_results = [
            self.resolve_conflicts_for_fact(fact, now=timestamp)
            for fact in record.facts
        ]
        saved = self.database.get_memory(memory_id)
        return {
            "operation": "created",
            "memory": saved,
            "facts": [
                {
                    "action": "add",
                    "fact_id": fact.id,
                    "target": fact.target,
                    "relation": fact.relation,
                }
                for fact in record.facts
            ]
            + reinforced,
            "conflicts": [item for item in conflict_results if item is not None],
        }

    def synchronize_memory_facts(
        self,
        existing_memory: Optional[Memory],
        new_memory: Optional[Memory],
    ) -> list[dict[str, Any]]:
        """Adaptador legado que agora anexa observações em vez de sobrescrevê-las."""

        if new_memory is None:
            return []
        result = self.remember(new_memory)
        return result.get("facts", [])

    def apply_memory_operation(
        self,
        reasoning_result,
        new_memory: Memory,
        existing_memory: Optional[Memory] = None,
    ):
        """Executa operações antigas com a semântica não destrutiva nova."""

        if reasoning_result is None:
            return "ignored"
        operation = getattr(reasoning_result, "memory_operation", "none")
        if operation in {"create", "add", "update", "supersede"}:
            result = self.remember(new_memory)
            if operation == "create" and result["operation"] == "created":
                return "created"
            return result.get("facts", result["operation"])
        if operation == "ignore":
            return "ignored"
        return "unknown"

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------

    def _fact_memory_type(self, fact: MemoryFact) -> str:
        if fact.fact_type:
            return fact.fact_type.casefold()
        memory = self.database.get_memory(fact.memory_id)
        return memory.memory_type.casefold() if memory else "fact"

    @staticmethod
    def _temporal_kind(temporal_context: Optional[str]) -> str:
        text = (temporal_context or "unknown").casefold()
        if any(word in text for word in ("past", "antes", "antigamente", "ontem", "hist")):
            return "past"
        if any(word in text for word in ("future", "futuro", "amanhã", "amanha", "depois")):
            return "future"
        if any(word in text for word in ("current", "agora", "atualmente", "hoje", "present")):
            return "current"
        return "unspecified"

    def _effective_polarity(self, fact: MemoryFact) -> int:
        relation = (fact.relation or "").casefold()
        if relation in self.POSITIVE_PREFERENCES:
            return -1 if fact.negation else 1
        if relation in self.NEGATIVE_PREFERENCES:
            # Bancos antigos representavam "não gosto" como dislike + negation.
            # Para não inverter esse histórico, relações negativas já codificadas
            # mantêm polaridade negativa; o valor bruto de negation continua salvo.
            return -1
        return -1 if fact.negation else 1

    def facts_conflict(self, left: MemoryFact, right: MemoryFact) -> bool:
        if left.id == right.id:
            return False
        if self.normalize_subject(left.subject) != self.normalize_subject(right.subject):
            return False
        if self.normalize_fact_target(left.target) != self.normalize_fact_target(right.target):
            return False
        if left.relation_family != right.relation_family:
            return False
        if "episodic" in {self._fact_memory_type(left), self._fact_memory_type(right)}:
            return False
        left_time = self._temporal_kind(left.temporal_context)
        right_time = self._temporal_kind(right.temporal_context)
        if left_time != right_time and {left_time, right_time} & {"past", "future"}:
            return False
        return self._effective_polarity(left) != self._effective_polarity(right)

    def _conflict_score(
        self,
        fact: MemoryFact,
        *,
        now: Optional[datetime | str] = None,
    ) -> float:
        evidence_count = len(fact.evidence or self.database.get_fact_evidence(fact.id))
        temporal_boost = {"current": 0.12, "unspecified": 0.05, "past": 0.0, "future": 0.0}[self._temporal_kind(fact.temporal_context)]
        return (
            fact.confidence * 0.55
            + self.effective_importance(fact, now=now) * 0.25
            + min(0.12, evidence_count * 0.03)
            + temporal_boost
        )

    def resolve_conflicts_for_fact(
        self,
        fact: MemoryFact,
        *,
        now: Optional[datetime | str] = None,
    ) -> Optional[dict[str, Any]]:
        """Persiste conflitos e aplica uma resolução determinística quando segura."""

        if fact.id is None:
            return None
        candidates = self.database.find_facts_for_semantic_key(
            fact.subject,
            fact.target,
            fact.relation_family,
            include_inactive=False,
            exclude_fact_id=fact.id,
        )
        resolutions = []
        for candidate in candidates:
            if not self.facts_conflict(candidate, fact):
                continue
            candidate_score = self._conflict_score(candidate, now=now)
            fact_score = self._conflict_score(fact, now=now)
            winner, loser = (
                (fact, candidate)
                if fact_score >= candidate_score
                else (candidate, fact)
            )
            difference = abs(fact_score - candidate_score)
            if difference >= 0.08:
                reason = (
                    "Conflito resolvido por confiança, evidências, importância "
                    "efetiva e contexto temporal."
                )
                conflict_id = self.database.record_conflict(
                    candidate.id,
                    fact.id,
                    status="resolved",
                    winner_fact_id=winner.id,
                    reason=reason,
                    now=now,
                )
                self.database.set_fact_status(
                    loser.id,
                    "superseded",
                    reason=f"Superado pelo fato #{winner.id}: {reason}",
                    now=now,
                )
                self.database._upsert_graph_edge(
                    f"fact:{winner.id}",
                    f"fact:{loser.id}",
                    "supersedes",
                    fact_id=winner.id,
                    weight=1.0,
                    metadata={"conflict_id": conflict_id},
                )
                self.database.connection.commit()
                resolutions.append(
                    {
                        "conflict_id": conflict_id,
                        "status": "resolved",
                        "winner_fact_id": winner.id,
                        "loser_fact_id": loser.id,
                    }
                )
            else:
                reason = "Conflito mantido aberto: evidências insuficientes para desempatar."
                conflict_id = self.database.record_conflict(
                    candidate.id,
                    fact.id,
                    status="open",
                    reason=reason,
                    now=now,
                )
                self.database.set_fact_status(
                    candidate.id,
                    "conflicted",
                    reason=reason,
                    now=now,
                )
                self.database.set_fact_status(
                    fact.id,
                    "conflicted",
                    reason=reason,
                    now=now,
                )
                resolutions.append(
                    {
                        "conflict_id": conflict_id,
                        "status": "open",
                        "winner_fact_id": None,
                    }
                )
        return {"fact_id": fact.id, "resolutions": resolutions} if resolutions else None

    # ------------------------------------------------------------------
    # Dynamic importance, forgetting and relevance retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    def _timestamp(self, value: Optional[datetime | str] = None) -> str:
        if isinstance(value, str):
            return value
        moment = value or self._now_provider()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat(timespec="seconds")

    def _now_datetime(self, value: Optional[datetime | str] = None) -> datetime:
        if isinstance(value, str):
            parsed = self._parse_timestamp(value)
            if parsed is not None:
                return parsed
        moment = value or self._now_provider()
        return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)

    def effective_importance(
        self,
        item: Memory | MemoryFact,
        *,
        now: Optional[datetime | str] = None,
    ) -> float:
        """Calcula importance dinâmica sem destruir o valor-base persistido."""

        if isinstance(item, MemoryFact):
            base = item.importance
            item_type = self._fact_memory_type(item)
            timestamp = item.updated_at or item.created_at
            confidence = item.confidence
            accesses = item.access_count
            evidence_count = len(item.evidence or self.database.get_fact_evidence(item.id))
            emotion_intensity = item.emotional_intensity
            status = item.status
        else:
            base = item.base_importance
            item_type = item.memory_type.casefold()
            timestamp = item.updated_at or item.created_at
            confidence = item.confidence
            accesses = item.access_count
            evidence_count = sum(len(fact.evidence) for fact in item.facts)
            emotion_intensity = item.emotional_intensity
            status = item.status

        created = self._parse_timestamp(timestamp) or self._now_datetime(now)
        age_days = max(0.0, (self._now_datetime(now) - created).total_seconds() / 86400)
        half_life = self.HALF_LIFE_DAYS.get(item_type, self.HALF_LIFE_DAYS["fact"])
        decay = 0.5 ** (age_days / half_life)
        confidence_factor = 0.55 + 0.45 * max(0.0, min(1.0, confidence))
        emotional_boost = min(0.10, abs(float(emotion_intensity or 0.0)) / 100)
        evidence_boost = min(0.12, evidence_count * 0.025)
        access_boost = min(0.10, math.log1p(max(0, accesses)) * 0.035)
        status_factor = {
            "active": 1.0,
            "conflicted": 0.72,
            "superseded": 0.45,
            "archived": 0.20,
            "retracted": 0.05,
        }.get(status, 0.6)
        score = base * decay * confidence_factor * status_factor
        return round(max(0.0, min(1.0, score + emotional_boost + evidence_boost + access_boost)), 6)

    def apply_decay(
        self,
        *,
        now: Optional[datetime | str] = None,
        archive_below: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Avalia decay; opcionalmente arquiva fatos muito fracos.

        Nada é apagado. Arquivamento é reversível e ganha uma revisão no
        histórico, preservando o mecanismo de esquecimento de forma auditável.
        """

        results: list[dict[str, Any]] = []
        for memory in self.database.find_memories(include_archived=True):
            for fact in memory.facts:
                score = self.effective_importance(fact, now=now)
                archived = False
                if (
                    archive_below is not None
                    and fact.status == "active"
                    and score < archive_below
                ):
                    archived = self.database.set_fact_status(
                        fact.id,
                        "archived",
                        reason=f"Arquivado por decay abaixo de {archive_below}.",
                        now=now,
                    )
                results.append(
                    {
                        "memory_id": memory.id,
                        "fact_id": fact.id,
                        "effective_importance": score,
                        "archived": archived,
                    }
                )
        return results

    def recall_relevant(
        self,
        query: str | Iterable[MemoryFact],
        *,
        limit: int = 10,
        include_inactive: bool = False,
        now: Optional[datetime | str] = None,
    ) -> list[dict[str, Any]]:
        """Recupera fatos ranqueados com uma explicação de relevância."""

        if isinstance(query, str):
            query_text = self.normalize_fact_target(query)
            query_targets = {query_text} if query_text else set()
            query_relations: set[str] = set()
        else:
            query_facts = list(query)
            query_targets = {
                self.normalize_fact_target(fact.target)
                for fact in query_facts
                if self.normalize_fact_target(fact.target)
            }
            query_relations = {
                (fact.relation or "").casefold() for fact in query_facts
            }
            query_text = " ".join(query_targets)
        query_tokens = set(query_text.split())

        ranked: list[dict[str, Any]] = []
        for memory in self.database.find_memories(include_archived=True):
            if memory.status == "archived" and not include_inactive:
                continue
            for fact in memory.facts:
                if not include_inactive and fact.status not in {"active", "conflicted"}:
                    continue
                target = self.normalize_fact_target(fact.target)
                target_tokens = set(target.split())
                reasons: list[str] = []
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
                temporal_kind = self._temporal_kind(fact.temporal_context)
                if temporal_kind == "current":
                    score += 0.05
                    reasons.append("contexto atual")
                elif temporal_kind == "past":
                    score += 0.01
                effective = self.effective_importance(fact, now=now)
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
        selected = ranked[: max(0, limit)]
        # Uma leitura reforça acesso, mas não altera updated_at nem a semântica.
        for memory_id in {item["memory"].id for item in selected}:
            self.database.touch_memory(memory_id, now=now)
        return selected

    def recall_relevant_facts(self, *args, **kwargs) -> list[MemoryFact]:
        return [item["fact"] for item in self.recall_relevant(*args, **kwargs)]
