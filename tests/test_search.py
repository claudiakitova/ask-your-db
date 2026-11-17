"""Search and the AI calls, without the internet: a tiny stand-in replaces the embedding model
and the AI service. Run: python -m unittest discover -s tests"""
import sys
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ai  # noqa: E402
import search  # noqa: E402


class WordCountModel:
    """Stands in for the embedding model: a vector of word counts, so shared words mean similar."""

    def vector(self, text):
        v = np.zeros(256)
        for w in search.words(text):
            v[zlib.crc32(w.encode()) % 256] += 1
        return v

    def embed(self, texts):
        return [self.vector(t) for t in texts]

    def query_embed(self, text):
        return iter([self.vector(text)])


class Passages(unittest.TestCase):
    def test_passages_end_at_sentences_and_overlap(self):
        text = " ".join(f"Sentence {i} has five words." for i in range(100))
        passages = search.split(text, size=50, overlap=10)
        self.assertTrue(all(len(p.split()) <= 60 for p in passages))
        self.assertTrue(all(p.endswith(".") for p in passages))
        self.assertIn(" ".join(passages[0].split()[-5:]), passages[1])  # the end of one starts the next

    def test_long_table_without_sentences_is_cut(self):
        table = "\n".join(f"| row {i} | {i * 10} |" for i in range(300))
        passages = search.split(table, size=100, overlap=0)
        self.assertGreater(len(passages), 5)
        self.assertIn("| row 299 | 2990 |", passages[-1])

    def test_nothing_is_lost(self):
        text = "First fact here. " * 40 + "The key is with Marta. " + "Last fact here. " * 40
        self.assertTrue(any("The key is with Marta." in p for p in search.split(text, size=30, overlap=5)))


class Ranking(unittest.TestCase):
    def test_bm25_prefers_the_exact_word(self):
        bm25 = search.BM25(["the invoice was paid", "invoice INV-4471 is overdue", "we like tea"])
        scores = bm25.scores("INV-4471")
        self.assertEqual(int(np.argmax(scores)), 1)
        self.assertEqual(scores[2], 0)

    def test_bm25_reads_russian(self):
        scores = search.BM25(["ключ у Марты", "погода хорошая"]).scores("Ключ")
        self.assertGreater(scores[0], scores[1])

    def test_fuse_rewards_both_lists(self):
        self.assertEqual(search.fuse([[3, 1, 2], [1, 4]])[0], 1)

    def test_index_finds_the_passage(self):
        parts = [{"text": "The garden opens on May 3.", "file": "plan.docx", "where": "Dates"},
                 {"text": "The greenhouse key is kept by Marta in the blue box.", "file": "notes.txt", "where": ""},
                 {"text": "Tomatoes need sun and water.", "file": "tips.pdf", "where": "p. 2"}]
        index = search.Index(parts, WordCountModel())
        found = index.search("Who keeps the greenhouse key?", top=2)
        self.assertEqual(found[0]["file"], "notes.txt")
        self.assertEqual(len(found), 2)


def reply(status=200, content="Marta.", text=""):
    response = mock.Mock(status_code=status, text=text)
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return response


class Services(unittest.TestCase):
    def test_answer_sends_the_passages(self):
        with mock.patch("ai.requests.post", return_value=reply()) as post:
            text = ai.answer("Groq", "openai/gpt-oss-20b", "k", [], "Who has the key?",
                             [{"text": "Marta has the key.", "file": "notes.txt", "where": "p. 1"}])
        self.assertEqual(text, "Marta.")
        sent = post.call_args.kwargs["json"]["messages"]
        self.assertIn("[notes.txt, p. 1]\nMarta has the key.", sent[0]["content"])
        self.assertEqual(sent[-1], {"role": "user", "content": "Who has the key?"})

    def test_reasoning_is_dropped(self):
        parts = [{"type": "thinking", "thinking": [{"type": "text", "text": "Let me think"}]},
                 {"type": "text", "text": "Marta."}]
        with mock.patch("ai.requests.post", return_value=reply(content=parts)):
            self.assertEqual(ai.ask("Mistral", "m", "k", []), "Marta.")

    def test_errors_are_explained(self):
        cases = [(401, "", "key didn't work"), (400, "Please pass a valid API key", "key didn't work"),
                 (429, "", "free limit"), (503, "high demand", "busy right now")]
        for status, body, expected in cases:
            with mock.patch("ai.requests.post", return_value=reply(status, text=body)):
                with self.assertRaisesRegex(ai.AIError, expected):
                    ai.ask("Gemini", "m", "k", [])

    def test_first_question_isnt_rewritten(self):
        with mock.patch("ai.requests.post") as post:
            self.assertEqual(ai.standalone("Gemini", "m", "k", [], "When does it open?"), "When does it open?")
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
