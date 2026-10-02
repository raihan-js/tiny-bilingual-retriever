import numpy as np
import pytest

from tinyrerank.eval import (build_relevance_judgments, compute_metrics,
                             cosine_search, dcg, ndcg_at_k, recall_at_k)


class TestDCG:
    def test_perfect_ranking(self):
        assert dcg(np.array([1, 1, 1]), 3) == pytest.approx(1.0 + 1/np.log2(3) + 1/np.log2(4))

    def test_empty(self):
        assert dcg(np.array([]), 10) == 0.0

    def test_single_relevant(self):
        assert dcg(np.array([1]), 10) == pytest.approx(1.0)


class TestNDCG:
    def test_perfect(self):
        rel = [np.array([1, 0, 0]), np.array([1, 1, 0])]
        assert ndcg_at_k(rel, 10) == pytest.approx(1.0)

    def test_imperfect(self):
        rel = [np.array([0, 1, 0])]
        assert 0 < ndcg_at_k(rel, 10) < 1

    def test_empty(self):
        assert ndcg_at_k([], 10) == 0.0


class TestRecall:
    def test_perfect(self):
        rel = [np.array([1, 0, 0, 0])]
        assert recall_at_k(rel, 1) == pytest.approx(1.0)

    def test_partial(self):
        rel = [np.array([0, 1, 0, 0])]
        assert recall_at_k(rel, 2) == pytest.approx(1.0)

    def test_missed(self):
        rel = [np.array([0, 0, 0, 1])]
        assert recall_at_k(rel, 2) == pytest.approx(0.0)

    def test_empty(self):
        assert recall_at_k([], 100) == 0.0


class TestCosineSearch:
    def test_finds_correct(self):
        q = np.array([[1.0, 0.0]])
        p = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        idx, scores = cosine_search(q, p, k=2)
        assert idx[0][0] == 0
        assert scores[0][0] == pytest.approx(1.0)

    def test_k_larger_than_passages(self):
        q = np.array([[1.0, 0.0]])
        p = np.array([[1.0, 0.0]])
        idx, scores = cosine_search(q, p, k=10)
        assert idx.shape == (1, 1)


class TestRelevanceJudgments:
    def test_builds_correctly(self):
        q_ids = ["q1", "q2"]
        ranked = np.array([[0, 1, 2], [2, 1, 0]])
        p_ids = ["p0", "p1", "p2"]
        rel = {"q1": {"p0"}, "q2": {"p2"}}
        judgments = build_relevance_judgments(q_ids, ranked, p_ids, rel)
        assert judgments[0][0] == 1.0
        assert judgments[0][1] == 0.0
        assert judgments[1][0] == 1.0


class TestComputeMetrics:
    def test_all_metrics(self):
        rel = [np.array([1, 0, 0]), np.array([0, 1, 0])]
        lat = [10.0, 20.0, 30.0, 40.0, 50.0]
        out = compute_metrics(rel, lat)
        assert "ndcg_at_10" in out
        assert "recall_at_100" in out
        assert "latency_p50_ms" in out
        assert "latency_p95_ms" in out
        assert out["latency_p50_ms"] == pytest.approx(30.0)