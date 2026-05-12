from unittest import TestCase

import numpy as np

from src.agents.zmat import get_angle, get_dihedral, get_distance, position_point


class TestDistance(TestCase):
    def test_zero(self):
        p = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(get_distance(p, p), 0.0)

    def test_unit(self):
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(get_distance(p0, p1), 1.0)

    def test_diagonal(self):
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 1.0, 0.0])
        self.assertAlmostEqual(get_distance(p0, p1), np.sqrt(2.0))


class TestAngle(TestCase):
    def test_zero(self):
        # p0 and p2 on the same side of p1 → angle = 0
        p0 = np.array([2.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([3.0, 0.0, 0.0])
        self.assertAlmostEqual(get_angle(p0, p1, p2), 0.0)

    def test_right_angle(self):
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([1.0, 1.0, 0.0])
        self.assertAlmostEqual(get_angle(p0, p1, p2), np.pi / 2)

    def test_straight(self):
        # p0-p1-p2 collinear with p0 and p2 on opposite sides of p1 → angle = π
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        self.assertAlmostEqual(get_angle(p0, p1, p2), np.pi)


class TestDihedral(TestCase):
    def setUp(self):
        self.p0 = np.array([0.0, 0.0, 0.0])
        self.p1 = np.array([0.0, 0.0, 1.0])
        self.p2 = np.array([0.0, 1.0, 1.0])

    def test_round_trip(self):
        for dihedral in np.linspace(-np.pi, np.pi, 100):
            p3 = position_point(self.p0, self.p1, self.p2, distance=1.0, angle=np.pi / 2, dihedral=dihedral)
            measured = get_dihedral(self.p0, self.p1, self.p2, p3)
            self.assertAlmostEqual(measured, dihedral, places=5)

    def test_round_trip_with_2pi_offset(self):
        # Skip ±π endpoints: -π and π represent the same angle, so get_dihedral
        # may return π when the input was -π.
        for dihedral in np.linspace(-np.pi, np.pi, 100)[1:-1]:
            p3 = position_point(self.p0, self.p1, self.p2, distance=1.0, angle=np.pi / 2, dihedral=dihedral + 2 * np.pi)
            measured = get_dihedral(self.p0, self.p1, self.p2, p3)
            self.assertAlmostEqual(measured, dihedral, places=5)

    def test_sign(self):
        p3_pos = position_point(self.p0, self.p1, self.p2, distance=1.0, angle=np.pi / 2, dihedral=np.pi / 2)
        p3_neg = position_point(self.p0, self.p1, self.p2, distance=1.0, angle=np.pi / 2, dihedral=-np.pi / 2)
        self.assertAlmostEqual(get_dihedral(self.p0, self.p1, self.p2, p3_pos), np.pi / 2, places=5)
        self.assertAlmostEqual(get_dihedral(self.p0, self.p1, self.p2, p3_neg), -np.pi / 2, places=5)

    def test_collinear_is_nan(self):
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.0, 0.0, 1.0])
        p2 = np.array([0.0, 0.0, 2.0])
        p3 = np.array([0.0, 0.0, 3.0])
        result = get_dihedral(p0, p1, p2, p3)
        self.assertTrue(np.isnan(result))


class TestPositionPoint(TestCase):
    def setUp(self):
        self.p0 = np.array([0.0, 0.0, 0.0])
        self.p1 = np.array([0.0, 0.0, 1.0])
        self.p2 = np.array([0.0, 1.0, 1.0])

    def test_distance_preserved(self):
        for distance in [0.5, 1.0, 1.5, 2.0]:
            p3 = position_point(self.p0, self.p1, self.p2, distance=distance, angle=np.pi / 3, dihedral=0.0)
            self.assertAlmostEqual(get_distance(self.p2, p3), distance, places=5)

    def test_angle_preserved(self):
        for angle in np.linspace(0.1, np.pi - 0.1, 20):
            p3 = position_point(self.p0, self.p1, self.p2, distance=1.0, angle=angle, dihedral=0.0)
            self.assertAlmostEqual(get_angle(self.p1, self.p2, p3), angle, places=5)

    def test_negative_angle_gives_absolute(self):
        for angle in np.linspace(-np.pi, 0.0, 50):
            p3 = position_point(self.p0, self.p1, self.p2, distance=1.0, angle=angle, dihedral=0.0)
            self.assertAlmostEqual(get_angle(self.p1, self.p2, p3), abs(angle), places=5)

    def test_negative_distance_preserves_length(self):
        p3_pos = position_point(self.p0, self.p1, self.p2, distance=1.0, angle=np.pi / 2, dihedral=0.0)
        p3_neg = position_point(self.p0, self.p1, self.p2, distance=-1.0, angle=np.pi / 2, dihedral=0.0)
        self.assertAlmostEqual(get_distance(self.p2, p3_pos), 1.0, places=5)
        self.assertAlmostEqual(get_distance(self.p2, p3_neg), 1.0, places=5)
