from unittest import TestCase

import numpy as np
import torch

from src.agents.modules import masked_softmax, to_one_hot


class TestOneHot(TestCase):
    def test_encoding(self):
        indices = torch.tensor([[1], [3], [2]])
        result = to_one_hot(indices, num_classes=4).detach().numpy()
        expected = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.float32)
        self.assertTrue(np.allclose(result, expected))

    def test_shape(self):
        indices = torch.tensor([[0], [1], [2], [3]])
        result = to_one_hot(indices, num_classes=5)
        self.assertEqual(result.shape, (4, 5))

    def test_out_of_range_raises(self):
        indices = torch.tensor([[5]])
        with self.assertRaises(RuntimeError):
            to_one_hot(indices, num_classes=3).detach()


class TestMaskedSoftmax(TestCase):
    def test_full_mask_sums_to_one_per_row(self):
        logits = torch.tensor([[0.5, 0.5], [1.0, 0.5]])
        mask = torch.ones(2, 2, dtype=torch.bool)
        result = masked_softmax(logits, mask)
        row_sums = result.sum(dim=1).detach().numpy()
        self.assertTrue(np.allclose(row_sums, [1.0, 1.0], atol=1e-5))

    def test_partial_mask_zeros_out_masked_column(self):
        logits = torch.tensor([[0.5, 0.5], [1.0, 0.5]])
        mask = torch.tensor([[1, 0], [1, 0]], dtype=torch.bool)
        result = masked_softmax(logits, mask)
        col_sums = result.sum(dim=0).detach().numpy()
        self.assertAlmostEqual(col_sums[0], 2.0, places=4)
        self.assertAlmostEqual(col_sums[1], 0.0, places=6)

    def test_none_mask_is_standard_softmax(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        expected = torch.nn.functional.softmax(logits, dim=-1)
        result = masked_softmax(logits, mask=None)
        self.assertTrue(torch.allclose(result, expected))
