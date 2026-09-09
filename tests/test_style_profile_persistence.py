import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_manager import MemoryManager


class StyleProfilePersistenceTests(unittest.TestCase):
    """Testes de persistencia do StyleProfile no SQLite."""

    def setUp(self):
        self.db_path = tempfile.mktemp(suffix=".db")

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_database_initializes_style_profile_table(self):
        mgr = MemoryManager(db_path=self.db_path)
        db = mgr.database
        db.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='style_profile'"
        )
        row = db.cursor.fetchone()
        self.assertIsNotNone(row, "Tabela style_profile nao foi criada")
        self.assertEqual(row["name"], "style_profile")

    def test_prompt_builder_injects_style(self):
        from brain.prompt_builder import PromptBuilder
        from brain.style_profile import StyleProfile

        profile = StyleProfile()
        profile.total_messages = 20
        profile.confidence = 1.0
        profile.formality = 0.2
        profile.abbreviation_ratio = 0.4
        profile.common_abbreviations = {"vc": 10, "tb": 8}
        profile.emoji_usage = 0.5

        builder = PromptBuilder()
        prompt = builder.build("oi", memories=[], style_profile=profile)

        self.assertIn("Estilo de fala", prompt.system)
        self.assertIn("informal", prompt.system)
        self.assertIn("vc", prompt.system)
        self.assertTrue(prompt.metadata.get("style_applied"))

    def test_prompt_builder_low_confidence_not_injected(self):
        from brain.prompt_builder import PromptBuilder
        from brain.style_profile import StyleProfile

        # Perfil com poucos dados nao deve injetar estilo
        profile = StyleProfile()
        profile.update("oi vc")
        profile.update("tb blz")

        builder = PromptBuilder()
        prompt = builder.build("oi", memories=[], style_profile=profile)
        self.assertNotIn("Estilo de fala", prompt.system)
        self.assertFalse(prompt.metadata.get("style_applied"))

    def test_prompt_builder_without_style_keeps_working(self):
        from brain.prompt_builder import PromptBuilder
        from brain.style_profile import StyleProfile

        # Perfil vazio nao deve quebrar nada
        empty = StyleProfile()
        builder = PromptBuilder()
        prompt = builder.build("oi", memories=[], style_profile=empty)
        self.assertNotIn("Estilo de fala", prompt.system)
        self.assertFalse(prompt.metadata.get("style_applied"))

        # Perfil None tambem nao quebra
        prompt2 = builder.build("oi", memories=[], style_profile=None)
        self.assertNotIn("Estilo de fala", prompt2.system)
        self.assertFalse(prompt2.metadata.get("style_applied"))

    def test_conversation_engine_updates_style(self):
        from brain.conversation_engine import ConversationEngine

        mgr = MemoryManager(db_path=self.db_path)
        engine = ConversationEngine(memory_manager=mgr)

        engine.process("oi vc blz")
        engine.process("tb to aqui pq preciso de ajuda")

        profile = mgr.style_profile
        self.assertGreaterEqual(profile.total_messages, 2)
        self.assertIn("vc", profile.common_abbreviations)
        self.assertIn("tb", profile.common_abbreviations)

    def test_style_flow_end_to_end(self):
        """Fluxo completo: mensagem -> perfil atualizado -> prompt com estilo."""
        from brain.conversation_engine import ConversationEngine
        from brain.style_profile import StyleProfile

        # Simula um usuario informal com muitas mensagens
        profile_msgs = [
            "oi vc blz", "tb to aqui", "pq vc nao responde",
            "vlw flw abs", "mt bom tb", "q isso", "e isso",
            "dmais", "mds q legal", "kk mt bom",
        ]
        profile = StyleProfile()
        for m in profile_msgs:
            profile.update(m)

        # Persiste via manager
        mgr = MemoryManager(db_path=self.db_path)
        import json
        mgr.database.save_style_profile(json.dumps(profile.to_dict()))

        # Engine carrega o perfil e injeta no prompt.
        # Passa um cognitive_core real para que prompt_builder exista.
        from brain.cognitive_core import CognitiveCore
        from brain.llm_provider import LLMProvider, LLMRequest, LLMResponse

        class _FakeLLM(LLMProvider):
            name = "fake"

            def initialize(self):
                return False

            def is_available(self):
                return False

            def generate(self, request):
                raise RuntimeError("nao deveria ser chamado")

            def unload(self):
                pass

        fake_core = CognitiveCore(llm_provider=_FakeLLM())
        engine = ConversationEngine(memory_manager=mgr, cognitive_core=fake_core)
        engine.process("oi")
        desc = mgr.style_profile.to_instruction()
        prompt = engine.cognitive_core.prompt_builder.build(
            "oi", memories=[], style_profile=mgr.style_profile
        )
        self.assertIn("Estilo de fala", prompt.system)
        self.assertIn(desc, prompt.system)

    def test_missing_profile_returns_default(self):
        mgr = MemoryManager(db_path=self.db_path)
        profile = mgr.style_profile
        self.assertEqual(profile.total_messages, 0)
        self.assertEqual(profile.confidence, 0.0)
        self.assertEqual(profile.formality, 0.5)
        self.assertEqual(profile.emoji_usage, 0.0)
        self.assertEqual(profile.avg_message_length, 50.0)
        self.assertEqual(profile.abbreviation_ratio, 0.0)
        self.assertEqual(profile.uses_punctuation, 0.8)
        self.assertEqual(profile.uses_caps, 0.0)
        self.assertEqual(profile.common_abbreviations, {})
        self.assertEqual(profile.common_words, {})
        self.assertEqual(profile.common_emojis, {})

    def test_style_profile_can_be_saved(self):
        mgr = MemoryManager(db_path=self.db_path)
        mgr.update_style_profile("vc e o melhor pq sempre ajuda tb")
        db = mgr.database
        raw = db.load_style_profile()
        self.assertIsNotNone(raw, "Perfil nao foi salvo no SQLite")
        import json
        data = json.loads(raw)
        self.assertIn("total_messages", data)
        self.assertGreater(data["total_messages"], 0)

    def test_style_profile_can_be_loaded(self):
        mgr = MemoryManager(db_path=self.db_path)
        mgr.update_style_profile("vc usa emoji e escreve mt rapido")
        mgr.update_style_profile("tmb gosto de pizza")
        mgr2 = MemoryManager(db_path=self.db_path)
        loaded = mgr2.style_profile
        self.assertEqual(loaded.total_messages, 2)
        self.assertIn("vc", loaded.common_abbreviations)
        self.assertIn("tmb", loaded.common_abbreviations)

    def test_all_fields_survive_roundtrip(self):
        mgr = MemoryManager(db_path=self.db_path)
        messages = [
            "vc e incrivel",
            "PQ VC NAO RESPONDE",
            "eu gosto muito de python tb",
            "Olá, tudo bem? Como voce esta?",
        ]
        for msg in messages:
            mgr.update_style_profile(msg)
        p_before = mgr.style_profile
        d_before = p_before.to_dict()
        mgr2 = MemoryManager(db_path=self.db_path)
        p_after = mgr2.style_profile
        d_after = p_after.to_dict()
        for key in d_before:
            self.assertEqual(d_before[key], d_after[key],
                f"Campo {key} nao sobreviveu ao roundtrip")

    def test_update_does_not_create_duplicates(self):
        mgr = MemoryManager(db_path=self.db_path)
        for i in range(5):
            mgr.update_style_profile(f"msg {i} vc tb")
        db = mgr.database
        db.cursor.execute("SELECT COUNT(*) as cnt FROM style_profile")
        count = db.cursor.fetchone()["cnt"]
        self.assertEqual(count, 1)
        mgr2 = MemoryManager(db_path=self.db_path)
        self.assertEqual(mgr2.style_profile.total_messages, 5)

    def test_existing_memory_operations_still_works(self):
        mgr = MemoryManager(db_path=self.db_path)
        from memory.memory import Memory
        from memory.memory_fact import MemoryFact
        m = Memory(
            content="eu gosto de pizza",
            memory_type="preference",
            importance=0.8,
            emotion="neutral",
            emotional_intensity=0.0,
            facts=[MemoryFact(target="pizza", relation="like")],
        )
        result = mgr.create_memory(m)
        self.assertIn("memory", result)
        self.assertIsNotNone(result["memory"].id)
        mgr.update_style_profile("vc gosta de pizza tb")
        memories = mgr.recall()
        self.assertGreaterEqual(len(memories), 1)

    def test_confidence_increases_with_messages(self):
        mgr = MemoryManager(db_path=self.db_path)
        initial = mgr.style_profile.confidence
        self.assertEqual(initial, 0.0)
        for i in range(15):
            mgr.update_style_profile(f"msg {i} vc tb pq")
        final = mgr.style_profile.confidence
        self.assertGreater(final, initial)
        self.assertLessEqual(final, 1.0)


if __name__ == "__main__":
    unittest.main()

