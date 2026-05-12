from unittest import TestCase

import numpy as np

from src.pretraining.action_decom import (
    bfs_sort_nodes,
    dfs_sort_nodes,
    gaussian_perturbation,
    recenter,
    rotate_to_axis,
)


# Simple 3-carbon chain: C-C-C at 1.5 Å spacing
CHAIN_ATOMS = [6, 6, 6]
CHAIN_POSITIONS = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])

# Methane-like: C at origin, 2 H atoms
CH2_ATOMS = [6, 1, 1]
CH2_POSITIONS = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-1.1, 0.0, 0.0]])


class TestGaussianPerturbation(TestCase):
    def test_zero_sigma_unchanged(self):
        pos = np.ones((5, 3))
        result = gaussian_perturbation(pos, sigma=0.0)
        self.assertTrue(np.allclose(result, pos))

    def test_none_sigma_unchanged(self):
        pos = np.ones((5, 3))
        result = gaussian_perturbation(pos, sigma=None)
        self.assertTrue(np.allclose(result, pos))

    def test_nonzero_sigma_changes_positions(self):
        np.random.seed(42)
        pos = np.ones((5, 3))
        result = gaussian_perturbation(pos, sigma=0.1)
        self.assertFalse(np.allclose(result, pos))

    def test_perturbation_shape_preserved(self):
        pos = np.zeros((4, 3))
        result = gaussian_perturbation(pos, sigma=0.05)
        self.assertEqual(result.shape, pos.shape)

    def test_perturbation_magnitude_is_small(self):
        # 10 sigma should capture virtually all perturbations
        np.random.seed(0)
        sigma = 0.05
        pos = np.zeros((100, 3))
        result = gaussian_perturbation(pos, sigma=sigma)
        self.assertTrue(np.all(np.abs(result) < 10 * sigma))


class TestRotateToAxis(TestCase):
    def test_atom_aligns_to_x_axis(self):
        positions = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
        rotated = rotate_to_axis(positions, atom_index=0, axis="x")
        self.assertAlmostEqual(rotated[0, 1], 0.0, places=8)
        self.assertAlmostEqual(rotated[0, 2], 0.0, places=8)

    def test_distance_preserved_after_rotation(self):
        positions = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        original_dist = np.linalg.norm(positions[0])
        rotated = rotate_to_axis(positions, atom_index=0, axis="x")
        rotated_dist = np.linalg.norm(rotated[0])
        self.assertAlmostEqual(rotated_dist, original_dist, places=8)

    def test_atom_aligns_to_y_axis(self):
        positions = np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 0.0]])
        rotated = rotate_to_axis(positions, atom_index=0, axis="y")
        self.assertAlmostEqual(rotated[0, 0], 0.0, places=8)
        self.assertAlmostEqual(rotated[0, 2], 0.0, places=8)

    def test_invalid_axis_raises(self):
        positions = np.array([[1.0, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            rotate_to_axis(positions, atom_index=0, axis="w")

    def test_invalid_atom_index_raises(self):
        positions = np.array([[1.0, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            rotate_to_axis(positions, atom_index=5, axis="x")

    def test_wrong_shape_raises(self):
        positions = np.array([[1.0, 0.0]])
        with self.assertRaises(ValueError):
            rotate_to_axis(positions, atom_index=0, axis="x")


class TestRecenter(TestCase):
    def test_heavy_first_centers_on_heaviest_atom(self):
        # CH2: C (z=6) is heaviest, must be only candidate
        pos = np.array([[2.0, 3.0, 1.0], [3.1, 3.0, 1.0], [0.9, 3.0, 1.0]])
        recentered = recenter(pos.copy(), CH2_ATOMS, mol_dataset="QM9", heavy_first=True)
        self.assertTrue(np.allclose(recentered[0], [0.0, 0.0, 0.0]))

    def test_relative_positions_preserved(self):
        pos = np.array([[1.0, 0.0, 0.0], [2.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        recentered = recenter(pos.copy(), CH2_ATOMS, mol_dataset="QM9", heavy_first=True)
        # Relative displacement between atom 1 and atom 0 should be unchanged
        original_diff = pos[1] - pos[0]
        recentered_diff = recentered[1] - recentered[0]
        self.assertTrue(np.allclose(original_diff, recentered_diff))

    def test_output_shape_unchanged(self):
        pos = CHAIN_POSITIONS.copy()
        recentered = recenter(pos, CHAIN_ATOMS, mol_dataset="QM9", heavy_first=True)
        self.assertEqual(recentered.shape, pos.shape)


class TestGraphTraversals(TestCase):
    def test_bfs_covers_all_atoms(self):
        result = bfs_sort_nodes(CHAIN_ATOMS, CHAIN_POSITIONS.copy(), hydrogen_delay=False)
        self.assertEqual(len(result), 3)
        self.assertEqual(set(result), {0, 1, 2})

    def test_dfs_covers_all_atoms(self):
        result = dfs_sort_nodes(CHAIN_ATOMS, CHAIN_POSITIONS.copy(), hydrogen_delay=False)
        self.assertEqual(len(result), 3)
        self.assertEqual(set(result), {0, 1, 2})

    def test_bfs_heavy_first_puts_carbon_first(self):
        result = bfs_sort_nodes(CH2_ATOMS, CH2_POSITIONS.copy(), hydrogen_delay=True)
        self.assertEqual(result[0], 0)  # C must come before H
        self.assertEqual(set(result), {0, 1, 2})
        self.assertEqual(len(result), 3)

    def test_dfs_heavy_first_puts_carbon_first(self):
        result = dfs_sort_nodes(CH2_ATOMS, CH2_POSITIONS.copy(), hydrogen_delay=True)
        self.assertEqual(result[0], 0)  # C must come before H
        self.assertEqual(set(result), {0, 1, 2})
        self.assertEqual(len(result), 3)

    def test_bfs_and_dfs_agree_on_simple_chain(self):
        # Linear chain with no branching: both traversals must produce same order
        bfs_result = bfs_sort_nodes(CHAIN_ATOMS, CHAIN_POSITIONS.copy(), hydrogen_delay=False)
        dfs_result = dfs_sort_nodes(CHAIN_ATOMS, CHAIN_POSITIONS.copy(), hydrogen_delay=False)
        self.assertEqual(list(bfs_result), list(dfs_result))
