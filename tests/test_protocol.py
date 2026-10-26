import unittest
import numpy as np
import torch
from arise.tags import weights
from arise.cwm import exclusive_ema
from arise.data import split_masks, pack, public_view
from arise.metrics import values, topk
from arise.models.main import TagLaunch as Adapter, CoupledWindowMixer as IndependentCWM
from arise.config import Config


class ProtocolTests(unittest.TestCase):
    def test_public_encoder_view(self):
        d = dict(
            round_index=np.arange(2),
            tags_clean=np.array([[1]]),
            address=np.array([[0]]),
            targets=np.ones((2, 5)),
            negatives=np.ones((2, 15)),
            item=np.arange(10),
            rating=np.ones(10),
        )
        self.assertEqual(set(public_view(d)), {"round_index", "tags_clean", "address"})

    def test_ordered_rank_and_empty(self):
        z = torch.tensor([[3, 2, 1, -1], [-1, -1, -1, -1]])
        (r, w) = weights(z, True)
        self.assertAlmostEqual(float(w[0].sum()), 0.7, places=6)
        self.assertTrue(w[0, 0] > w[0, 1] > w[0, 2])
        self.assertEqual(float(w[1].sum()), 0)
        self.assertEqual(float(r[1]), 1)

    def test_unordered_permutation_and_multitype(self):
        emb = torch.randn(6, 64)
        z = torch.tensor([[0, 1, 4, -1]])
        zz = torch.tensor([[4, 0, 1, -1]])
        (_, a) = weights(z, False)
        (_, b) = weights(zz, False)
        torch.testing.assert_close(
            (a[:, :, None] * emb[z.clamp_min(0)]).sum(1),
            (b[:, :, None] * emb[zz.clamp_min(0)]).sum(1),
        )
        (_, second) = weights(torch.tensor([[2, -1]]), True)
        self.assertAlmostEqual(float(a.sum()), float(second.sum()), places=6)

    def test_adapter_identity_initial_and_empty(self):
        a = Adapter(64)
        x = torch.randn(3, 64)
        self.assertEqual(float(a(x).abs().max()), 0)
        a(x).sum().backward()
        self.assertGreater(float(a[2].weight.grad.abs().sum()), 0)
        self.assertEqual(float(a(torch.zeros(1, 64)).abs().max()), 0)

    def test_independent_gates(self):
        g = IndependentCWM(Config())
        w = g.weights(torch.randn(3, 12), torch.tensor([True, False, False]))
        self.assertEqual(tuple(w.shape), (3, 2))
        self.assertEqual(float(w[0].sum()), 0)
        torch.testing.assert_close(w[1:], torch.full((2, 2), 0.1))

    def test_exclusive_evidence(self):
        gain = torch.randn(410, 3)
        owner = torch.repeat_interleave(torch.arange(2), 205)
        a = exclusive_ema(gain, owner)
        expected = torch.zeros_like(a)
        for u in range(2):
            carry = torch.zeros(3)
            for j in range(u * 205, (u + 1) * 205):
                expected[j] = carry
                carry = 0.8 * carry + 0.2 * gain[j]
        torch.testing.assert_close(a, expected, atol=1e-06, rtol=1e-05)
        edited = gain.clone()
        edited[20:205] += 100
        b = exclusive_ema(edited, owner)
        torch.testing.assert_close(a[:21], b[:21])
        torch.testing.assert_close(a[205:], b[205:])

    def test_group_prefix_no_current_feedback(self):
        d = dict(
            offsets=np.array([0, 15]),
            round_user=np.zeros(3, int),
            round_index=np.arange(3),
            item=np.arange(15),
            timestamp=np.arange(15),
            round_start=np.array([0, 5, 10]),
            round_end=np.array([4, 9, 14]),
        )
        p = pack(d, np.array([0]))
        self.assertEqual(p["lengths"].tolist(), [0, 5, 10])
        self.assertEqual(p["history"][1, :5].tolist(), [0, 1, 2, 3, 4])
        d["item"][5:10] = 99
        q = pack(d, np.array([0]))
        np.testing.assert_array_equal(q["history"][:2], p["history"][:2])

    def test_split_crossing_guard(self):
        ts = np.array(
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 100, 101],
                [10, 11, 12, 13, 14],
                [102, 103, 104, 105, 106],
            ]
        )
        (tr, va, cut) = split_masks(ts)
        self.assertTrue(np.all(ts[tr, -1] < cut))
        self.assertTrue(np.all(ts[va, 0] >= cut))
        self.assertFalse((tr & va).any())

    def test_unique_target_metrics_and_unscorable(self):
        cat = torch.arange(25)
        p = torch.arange(25)
        scores = torch.zeros(1, 25)
        scores[:, 0] = -torch.inf
        t = topk(scores, cat, p)
        self.assertNotIn(0, t.tolist()[0])
        v = values(t, torch.tensor([[1, 1, 2, 3, 4]]))
        self.assertAlmostEqual(float(v[0, 1]), 1.0)
        self.assertAlmostEqual(float(v[0, 2]), 0.8)
        s = torch.full((1, 25), -torch.inf)
        self.assertTrue((topk(s, cat, p) == -1).all())


if __name__ == "__main__":
    unittest.main()
