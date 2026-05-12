from unittest import TestCase

import numpy as np

from src.rl.buffer import (
    DynamicPPOBuffer,
    collect_data_batch,
    get_batch_generator,
)


def _make_dummy_obs(seed: int = 0):
    return (((seed, (0.0, 0.0, 0.0)),), (0, 1))


class TestDynamicPPOBuffer(TestCase):
    def _fill_trajectory(self, buf: DynamicPPOBuffer, n_steps: int, reward: float = 1.0):
        for i in range(n_steps):
            buf.store(
                obs=_make_dummy_obs(i),
                act=np.array([0, 0.0, 0.0, 0.0]),
                reward=reward,
                next_obs=_make_dummy_obs(i + 1),
                terminal=(i == n_steps - 1),
                value=0.5,
                logp=-1.0,
            )

    def test_initial_state_is_not_finished(self):
        buf = DynamicPPOBuffer()
        self.assertTrue(buf.is_finished())

    def test_store_increments_index(self):
        buf = DynamicPPOBuffer()
        self._fill_trajectory(buf, 3)
        self.assertEqual(buf.current_index, 3)
        self.assertFalse(buf.is_finished())

    def test_finish_path_returns_episodic_return_and_length(self):
        buf = DynamicPPOBuffer(gamma=1.0, lam=1.0)
        self._fill_trajectory(buf, 3, reward=1.0)
        ep_return, ep_length = buf.finish_path(last_val=0.0)
        # With gamma=1, undiscounted sum = 3.0
        self.assertAlmostEqual(ep_return, 3.0, places=5)
        self.assertEqual(ep_length, 3)

    def test_finish_path_marks_buffer_finished(self):
        buf = DynamicPPOBuffer()
        self._fill_trajectory(buf, 4)
        self.assertFalse(buf.is_finished())
        buf.finish_path(last_val=0.0)
        self.assertTrue(buf.is_finished())

    def test_gae_advantage_calculation(self):
        # Manual check: gamma=1, lam=1, 2 steps, r=[1,2], v=[3,4], last_val=0
        # deltas = [1+4-3, 2+0-4] = [2, -2]
        # adv = [2 + -2, -2] = [0, -2]
        # ret = [3, 2]
        buf = DynamicPPOBuffer(gamma=1.0, lam=1.0)
        buf.store(_make_dummy_obs(), np.zeros(4), 1.0, _make_dummy_obs(), False, 3.0, -1.0)
        buf.store(_make_dummy_obs(), np.zeros(4), 2.0, _make_dummy_obs(), True, 4.0, -1.0)
        buf.finish_path(last_val=0.0)

        self.assertAlmostEqual(buf.adv_buf[0], 0.0, places=5)
        self.assertAlmostEqual(buf.adv_buf[1], -2.0, places=5)
        self.assertAlmostEqual(buf.ret_buf[0], 3.0, places=5)
        self.assertAlmostEqual(buf.ret_buf[1], 2.0, places=5)

    def test_get_data_normalizes_advantages(self):
        buf = DynamicPPOBuffer(gamma=0.99, lam=0.95)
        self._fill_trajectory(buf, 5, reward=1.0)
        buf.finish_path(last_val=0.0)
        data = buf.get_data()

        adv = data["adv"]
        self.assertAlmostEqual(np.mean(adv), 0.0, places=5)
        self.assertAlmostEqual(np.std(adv), 1.0, places=5)

    def test_get_data_shapes(self):
        buf = DynamicPPOBuffer()
        n = 6
        self._fill_trajectory(buf, n)
        buf.finish_path(last_val=0.0)
        data = buf.get_data()

        self.assertEqual(len(data["obs"]), n)
        self.assertEqual(data["act"].shape, (n, 4))
        self.assertEqual(data["ret"].shape, (n,))
        self.assertEqual(data["adv"].shape, (n,))
        self.assertEqual(data["logp"].shape, (n,))

    def test_two_episodes_accumulate(self):
        buf = DynamicPPOBuffer(gamma=1.0, lam=1.0)
        self._fill_trajectory(buf, 3, reward=1.0)
        buf.finish_path(last_val=0.0)
        self._fill_trajectory(buf, 2, reward=2.0)
        buf.finish_path(last_val=0.0)

        self.assertEqual(buf.current_index, 5)
        self.assertTrue(buf.is_finished())


class TestGetBatchGenerator(TestCase):
    def test_all_indices_covered(self):
        indices = np.arange(10)
        all_batches = list(get_batch_generator(indices, batch_size=3))
        all_seen = np.concatenate(all_batches)
        self.assertEqual(sorted(all_seen), list(range(10)))

    def test_batch_sizes(self):
        indices = np.arange(10)
        batches = list(get_batch_generator(indices, batch_size=3))
        # Batches from divisible part: floor(10/3) = 3 batches of 3 = 9, remainder 1
        full_batches = [b for b in batches if len(b) == 3]
        self.assertEqual(len(full_batches), 3)
        remainder = [b for b in batches if len(b) < 3]
        self.assertEqual(len(remainder), 1)
        self.assertEqual(len(remainder[0]), 1)

    def test_exact_divisor_no_remainder(self):
        indices = np.arange(9)
        batches = list(get_batch_generator(indices, batch_size=3))
        self.assertEqual(len(batches), 3)
        self.assertTrue(all(len(b) == 3 for b in batches))


class TestCollectDataBatch(TestCase):
    def test_numpy_arrays(self):
        data = {"a": np.array([10, 20, 30, 40, 50])}
        batch = collect_data_batch(data, indices=np.array([1, 3]))
        self.assertTrue(np.array_equal(batch["a"], [20, 40]))

    def test_list_values(self):
        data = {"obs": ["a", "b", "c", "d"]}
        batch = collect_data_batch(data, indices=np.array([0, 2]))
        self.assertEqual(batch["obs"], ["a", "c"])

    def test_mixed_keys(self):
        data = {
            "arr": np.array([1.0, 2.0, 3.0]),
            "lst": ["x", "y", "z"],
        }
        batch = collect_data_batch(data, indices=np.array([2]))
        self.assertAlmostEqual(batch["arr"][0], 3.0)
        self.assertEqual(batch["lst"], ["z"])
