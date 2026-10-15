"""Pieces of the FlashMap sglang draft that run without a server: the
multi-jump noise bank (pure torch), the draft-vocabulary mapping of the eager
and folded samplers, and the fused sampling kernel (CUDA)."""

import unittest

import torch
import torch.nn.functional as F

from flashmap.sglang.jump_bank import (
    JumpNoiseBank,
    jump_coefficients,
    token_normals,
)

try:
    from flashmap.sglang.sampler import IdentityMarkov, _FullLogits
except Exception:  # sglang (0.5.20) not installed
    IdentityMarkov = None


class _Draft:
    """The slice of FlashMapDraftModel the bank reads, on a tiny vocabulary."""

    def __init__(self, vocab=40, hidden=8):
        generator = torch.Generator().manual_seed(0)
        self.embedding = torch.randn(vocab, hidden, generator=generator)
        self.gate = torch.ones(1)

    def state_norm(self, x):
        return F.rms_norm(x, (x.shape[-1],), eps=1e-6)

    noise_norm = state_norm

    def noise_gate_at(self, start, end):
        return self.gate * (end - start)


class TestJumpBank(unittest.TestCase):
    grid = [0.0, 0.8172569274902344, 1.0]

    def test_coefficients(self):
        (alpha0, betas0), (alpha1, betas1) = jump_coefficients(self.grid)
        self.assertEqual((alpha0, betas0), (1.0, []))
        self.assertAlmostEqual(alpha1, 1.0 - self.grid[1])
        self.assertEqual(len(betas1), 1)
        self.assertAlmostEqual(betas1[0], self.grid[1])

    def test_token_normals_are_a_function_of_seed_and_token(self):
        seeds = torch.arange(4096)
        ids = torch.randint(0, 151936, (4096, 16))
        first, again = token_normals(seeds, ids), token_normals(seeds, ids)
        self.assertTrue(torch.equal(first, again))
        self.assertLess(abs(first.mean().item()), 0.02)
        self.assertLess(abs(first.std().item() - 1.0), 0.02)

    def test_jump_input_is_exact_without_predictions(self):
        # With all-zero predictions the support terms vanish, so jump 1's input
        # must equal the literal embedding of x_1 = alpha_1 z over the vocabulary.
        draft = _Draft()
        bank = JumpNoiseBank(draft, self.grid, size=8, temperature=1.0, seed=3, chunk=4)
        generator = torch.Generator().manual_seed(3)
        z = torch.cat([torch.randn(4, 40, generator=generator) for _ in range(2)])
        rows = torch.tensor([0, 5, 7])
        seeds = torch.tensor([1, 2, 3])
        ids = torch.zeros(3, 4, dtype=torch.int64)
        supports = [(ids, torch.zeros(3, 4))]
        banked = bank.jump_input(draft, 1, rows, seeds, supports)
        t, (alpha, _) = self.grid[1], jump_coefficients(self.grid)[1]
        state = alpha * z[rows]
        posterior = (t / (1 - t) ** 2 * state).softmax(-1) @ draft.embedding
        literal = draft.state_norm(posterior) + draft.noise_gate_at(
            t, 1.0
        ) * draft.noise_norm(state @ draft.embedding)
        torch.testing.assert_close(banked, literal, rtol=1e-4, atol=1e-4)


@unittest.skipIf(IdentityMarkov is None, "needs sglang 0.5.20")
class TestDraftVocabulary(unittest.TestCase):
    hot_ids = torch.tensor([1, 4, 7])

    def test_eager_sampler_maps_tokens_and_scatters_logits(self):
        head = IdentityMarkov(vocab_size=10)
        head.hot_ids = self.hot_ids
        logits = torch.tensor([[[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]]])
        tokens, full = head.sample_block(
            logits,
            first_prev_tokens=None,
            hidden_states=None,
            sampler=lambda step_logits, step: step_logits.argmax(-1),
        )
        self.assertEqual(tokens.tolist(), [[4, 1]])
        self.assertEqual(full.shape, (1, 2, 10))
        self.assertTrue(torch.equal(full[..., self.hot_ids], logits))
        others = torch.ones(10, dtype=torch.bool)
        others[self.hot_ids] = False
        self.assertTrue(torch.isneginf(full[..., others]).all())

    def test_folded_logits_scatter_on_read(self):
        full = torch.zeros(4, 10)
        hot = torch.arange(12, dtype=torch.float32).view(4, 3)
        lazy = _FullLogits(full, hot, self.hot_ids)
        rows = lazy[:2]
        self.assertTrue(torch.equal(rows[:, self.hot_ids], hot[:2]))
        self.assertTrue(torch.isneginf(rows[:, 0]).all())
        self.assertTrue(torch.isneginf(full[2:]).all())  # rows past the read stay -inf


@unittest.skipIf(not torch.cuda.is_available(), "needs CUDA")
class TestSampleRows(unittest.TestCase):
    def test_greedy_rows_argmax_and_sampled_rows_follow_softmax(self):
        from flashmap.sglang.kernels import sample_rows

        rows, vocab = 4096, 5000
        logits = torch.randn(rows, vocab, device="cuda").to(torch.bfloat16)
        logits[:, :4] = torch.tensor([3.0, 2.0, 1.0, 0.0], device="cuda").to(
            torch.bfloat16
        )
        logits[:, 4:] -= 30  # four live tokens
        greedy = torch.zeros(rows, dtype=torch.bool, device="cuda")
        greedy[:64] = True
        temperatures = torch.full((rows,), 2.0, device="cuda")
        seed = torch.zeros((), dtype=torch.int64, device="cuda")
        tokens = sample_rows(logits, temperatures, greedy, seed)
        self.assertTrue(torch.equal(tokens[:64], logits[:64].float().argmax(-1)))
        counts = torch.bincount(tokens[64:], minlength=vocab)[:4].float()
        expected = torch.softmax(torch.tensor([3.0, 2.0, 1.0, 0.0]) / 2.0, 0)
        torch.testing.assert_close(
            counts.cpu() / counts.sum().cpu(), expected, atol=0.03, rtol=0
        )
        seed.add_(1)
        self.assertFalse(
            torch.equal(sample_rows(logits, temperatures, greedy, seed), tokens)
        )


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    "needs FP8 tensor cores",
)
class TestFp8Linear(unittest.TestCase):
    def test_close_to_bf16_linear(self):
        from flashmap.sglang.kernels import (
            fp8_linear,
            quantize_weight_fp8,
        )

        torch.manual_seed(0)
        x = torch.randn(48, 2048, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(6144, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
        bias = torch.randn(6144, device="cuda", dtype=torch.bfloat16)
        reference = F.linear(x, weight, bias).float()
        got = fp8_linear(x, *quantize_weight_fp8(weight), bias).float()
        error = (got - reference).norm() / reference.norm()
        self.assertLess(error.item(), 0.05)


@unittest.skipIf(not torch.cuda.is_available(), "needs CUDA")
class TestFp8KvWrite(unittest.TestCase):
    def test_matches_sglang_bf16_write(self):
        try:
            from sglang.kernels.ops.speculative.dspark.fused_kv_write import (
                fused_kv_norm_rope_write,
            )
        except Exception:
            self.skipTest("needs sglang 0.5.20")
        from flashmap.sglang.kernels import kv_norm_rope_write_fp8

        torch.manual_seed(0)
        layers, heads, dim, slots, rows = 2, 4, 128, 64, 10
        kv = torch.randn(rows, layers * 2 * heads * dim, device="cuda").to(
            torch.bfloat16
        )
        norm = torch.rand(layers, dim, device="cuda") + 0.5
        angles = torch.randn(256, dim // 2, device="cuda")
        cos_sin = torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()
        positions = torch.randint(0, 256, (rows,), device="cuda")
        locs = torch.randperm(slots - 1, device="cuda")[:rows] + 1
        pools = {}
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            buffers = [
                torch.zeros(slots, heads, dim, device="cuda").to(dtype)
                for _ in range(2 * layers)
            ]
            meta = torch.tensor(
                [
                    [
                        buffers[2 * i].data_ptr(),
                        buffers[2 * i + 1].data_ptr(),
                        heads * dim,
                        heads * dim,
                    ]
                    for i in range(layers)
                ],
                dtype=torch.int64,
                device="cuda",
            )
            write = (
                kv_norm_rope_write_fp8
                if dtype == torch.float8_e4m3fn
                else lambda *a, **k: fused_kv_norm_rope_write(
                    *a, is_neox_style=False, **k
                )
            )
            write(
                kv, meta, norm, cos_sin, positions, locs, layers, heads * dim, dim, 1e-5
            )
            pools[dtype] = buffers
        for bf16, fp8 in zip(pools[torch.bfloat16], pools[torch.float8_e4m3fn]):
            torch.testing.assert_close(
                fp8[locs].float(), bf16[locs].float(), rtol=0.07, atol=0.02
            )


class TestBuildTree(unittest.TestCase):
    def test_trie_of_candidates(self):
        from flashmap.sglang.tree import build_tree_abi

        # Two requests, three candidates of four drafts each; width 6 keeps 5 nodes.
        candidates = torch.tensor(
            [
                [[1, 2, 3, 4], [1, 2, 5, 6], [7, 8, 9, 10]],
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            ]
        )
        width = 6
        tokens, ids, stats = build_tree_abi(candidates, width)
        self.assertEqual(stats.tolist(), [[10, 3], [4, 1]])
        # Request 0: the shared prefix [1, 2] (votes 2) first, then the rest by
        # depth: 1, 7 at depth 0; 2, 8 at depth 1; 3 at depth 2.
        self.assertEqual(tokens[0].tolist(), [1, 7, 2, 8, 3])
        parents = (ids // width).tolist()
        self.assertEqual(parents[0], [0, 0, 1, 2, 3])
        # Request 1: one chain of four nodes, padded with a copy of the first.
        self.assertEqual(tokens[1].tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(parents[1], [0, 1, 2, 3, 0])


class TestTreeRefusal(unittest.TestCase):
    """A tree-mode server refuses what its greedy tree verify cannot serve."""

    @staticmethod
    def _req(top_k=1, return_logprob=False, **constraints):
        from types import SimpleNamespace

        params = SimpleNamespace(
            top_k=top_k, json_schema=None, regex=None, ebnf=None, structural_tag=None
        )
        params.__dict__.update(constraints)
        return SimpleNamespace(sampling_params=params, return_logprob=return_logprob)

    def test_greedy_served_others_refused(self):
        from flashmap.sglang.tree_worker import tree_refusal

        self.assertIsNone(tree_refusal(self._req()))
        self.assertIn("sampling", tree_refusal(self._req(top_k=1 << 30)))
        self.assertIn("logprobs", tree_refusal(self._req(return_logprob=True)))
        self.assertIn("constrained", tree_refusal(self._req(regex="a+")))
        both = tree_refusal(self._req(top_k=20, json_schema="{}"))
        self.assertIn("sampling (temperature > 0) and constrained", both)
