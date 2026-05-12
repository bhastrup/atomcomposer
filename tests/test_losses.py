from unittest import TestCase

from src.rl.losses import EntropySchedule, RewardCoefficientSchedule


class TestEntropySchedule(TestCase):
    def setUp(self):
        # Ramp from 0.5 down to 0.1 between iterations 100 and 200
        self.schedule = EntropySchedule(start_value=0.5, final_value=0.1, start_iter=100, end_iter=200)

    def test_before_ramp_returns_start(self):
        self.assertAlmostEqual(self.schedule.calculate(0), 0.5)
        self.assertAlmostEqual(self.schedule.calculate(100), 0.5)

    def test_after_ramp_returns_final(self):
        self.assertAlmostEqual(self.schedule.calculate(200), 0.1)
        self.assertAlmostEqual(self.schedule.calculate(999), 0.1)

    def test_midpoint(self):
        # At step 150 (halfway), value = 0.5 + 0.5*(0.1 - 0.5) = 0.3
        self.assertAlmostEqual(self.schedule.calculate(150), 0.3)

    def test_monotonically_decreasing(self):
        steps = range(100, 201, 10)
        values = [self.schedule.calculate(s) for s in steps]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_ramp_up(self):
        schedule = EntropySchedule(start_value=0.0, final_value=1.0, start_iter=0, end_iter=100)
        self.assertAlmostEqual(schedule.calculate(0), 0.0)
        self.assertAlmostEqual(schedule.calculate(50), 0.5)
        self.assertAlmostEqual(schedule.calculate(100), 1.0)


class TestRewardCoefficientSchedule(TestCase):
    def setUp(self):
        self.schedule = RewardCoefficientSchedule(
            schedules={
                "rew_dipole": (0.0, 2.0),
                "rew_basin": (1.0, 0.0),
            },
            start_iter=0,
            end_iter=100,
        )

    def test_returns_all_keys(self):
        result = self.schedule.calculate(50)
        self.assertIn("rew_dipole", result)
        self.assertIn("rew_basin", result)

    def test_start_values(self):
        result = self.schedule.calculate(0)
        self.assertAlmostEqual(result["rew_dipole"], 0.0)
        self.assertAlmostEqual(result["rew_basin"], 1.0)

    def test_final_values(self):
        result = self.schedule.calculate(100)
        self.assertAlmostEqual(result["rew_dipole"], 2.0)
        self.assertAlmostEqual(result["rew_basin"], 0.0)

    def test_midpoint_values(self):
        result = self.schedule.calculate(50)
        self.assertAlmostEqual(result["rew_dipole"], 1.0)
        self.assertAlmostEqual(result["rew_basin"], 0.5)

    def test_clamps_before_start(self):
        result = self.schedule.calculate(-10)
        self.assertAlmostEqual(result["rew_dipole"], 0.0)

    def test_clamps_after_end(self):
        result = self.schedule.calculate(200)
        self.assertAlmostEqual(result["rew_dipole"], 2.0)
