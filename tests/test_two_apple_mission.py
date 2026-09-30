import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grid_nav"))
from webots_adapter import MissionVisits, ProgressWatchdog, TARGET_COUNT
from motion_control import MotionController
from dwa import dwa_control, DWAParams, simulate_trajectories


class MissionTests(unittest.TestCase):
    def test_two_distinct_confirmed_visits_required(self):
        m = MissionVisits()
        targets = [(1., 0.), (4., 0.)]
        self.assertEqual(TARGET_COUNT, 2)
        self.assertEqual(m.select(targets, (0., 0., 0.), 0), 0)
        self.assertFalse(m.finish(.3, False))
        self.assertFalse(m.finish(float('nan'), True))
        self.assertFalse(m.finish(.7, True))
        self.assertTrue(m.finish(.3, True))
        self.assertFalse(m.complete)
        targets[0] = (1.2, .1)  # detector refines location, identity must not change
        self.assertEqual(m.select(targets, (0., 0., 0.), 1), 1)
        self.assertTrue(m.finish(.3, True))
        self.assertTrue(m.complete)
        self.assertIsNone(m.select(targets, (0., 0., 0.), 2))

    def test_failure_is_not_a_visit_and_can_retry(self):
        m = MissionVisits()
        m.select([(1., 0.)], (0., 0., 0.), 0)
        m.defer(0)
        self.assertEqual(len(m.completed), 0)
        self.assertIsNone(m.select([(1., 0.)], (0., 0., 0.), 1))
        self.assertEqual(m.select([(1., 0.)], (0., 0., 0.), 21), 0)

    def test_one_apple_cannot_be_counted_twice(self):
        m = MissionVisits()
        m.select([(1., 0.)], (0., 0., 0.), 0)
        m.finish(.3, True)
        self.assertIsNone(m.select([(1.1, 0.)], (0., 0., 0.), 1))
        self.assertFalse(m.finish(.3, True))
        self.assertFalse(m.complete)

    def test_recovery_reverse_checks_rear_obstacle(self):
        c = MotionController()
        angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        ranges = np.full(360, np.inf)
        ranges[0] = .14
        self.assertEqual(c.guard_command(-.1, 0., ranges, angles), (0., 0.))
        self.assertEqual(c.guard_command(.1, 0., ranges, angles), (.1, 0.))
        self.assertEqual(c.guard_command(.1, 0., np.full(360, np.nan), angles), (0., 0.))

    def test_stationary_dwa_is_detected_without_forward_command(self):
        w = ProgressWatchdog()
        self.assertFalse(w.update((0., 0., 0.), 0, True))
        self.assertTrue(w.update((.01, 0., 2.), 16, True))
        self.assertFalse(w.update((.2, 0., 2.), 17, True))
        self.assertFalse(w.update((.2, 0., 2.), 40, False))
        self.assertFalse(w.update((.2, 0., 2.), 41, True))

    def test_low_obstacles_remain_in_local_planner(self):
        c = MotionController()
        c.remember_obstacle(.25, 0., .15)
        c.remember_obstacle(.25, 0., .15)
        self.assertEqual(len(c._virtual_obstacles), 1)
        self.assertGreater(len(c._remembered_points((0., 0., 0.))), 0)
        self.assertEqual(len(c._remembered_points((10., 0., 0.))), 0)

    def test_emergency_candidate_must_not_approach_obstacle(self):
        params = DWAParams()
        obstacle = np.array([[.18, 0.]])
        (v, w), debug = dwa_control((0., 0., 0.), 0., 0., (1., 0.), obstacle,
                                    params, return_debug=True)
        if abs(v) > 1e-8:
            trajectory = simulate_trajectories((0., 0., 0.), np.array([v]), np.array([w]), params)
            self.assertGreaterEqual(np.linalg.norm(trajectory[0, :, :2] - obstacle, axis=1).min(), .178)


if __name__ == '__main__':
    unittest.main()
