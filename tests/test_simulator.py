"""Tests for RobotSwarmSimulator."""
import math

import numpy as np
import pytest

from swarm.robot_swarm_simulator import RobotSwarmSimulator


@pytest.fixture
def sim():
    return RobotSwarmSimulator(num_robots=4, noise_scale=0.5, seed=42)


# ── distances ────────────────────────────────────────────────────────────────

def test_distances_non_negative(sim):
    d = sim.get_noisy_distances()
    for i, row in d.items():
        for j, v in row.items():
            assert v >= 0.0, f"negative distance d[{i}][{j}] = {v}"


def test_no_self_distance(sim):
    d = sim.get_noisy_distances()
    for i in sim.robot_ids:
        assert i not in d[i]


def test_distances_all_pairs_present(sim):
    d = sim.get_noisy_distances()
    for i in sim.robot_ids:
        for j in sim.robot_ids:
            if i != j:
                assert j in d[i], f"missing d[{i}][{j}]"


def test_distances_close_to_truth(sim):
    """Noisy measurements should not deviate from truth by more than 3× noise_scale."""
    d = sim.get_noisy_distances()
    for i in sim.robot_ids:
        for j, v in d[i].items():
            true_d = sim._true_distance(i, j)
            assert abs(v - true_d) <= 3 * sim.noise_scale + true_d * 0.01


def test_comm_range_filters_distant_pairs():
    sim = RobotSwarmSimulator(num_robots=4, seed=0)
    # place robots far apart so max_comm_range cuts them off
    sim.true_positions = {
        0: np.array([0.0, 0.0]),
        1: np.array([100.0, 0.0]),
        2: np.array([0.0, 100.0]),
        3: np.array([100.0, 100.0]),
    }
    sim.max_comm_range = 5.0
    d = sim.get_noisy_distances()
    for i in sim.robot_ids:
        assert len(d[i]) == 0, f"robot {i} should have no neighbours within range"


# ── bearings ─────────────────────────────────────────────────────────────────

def test_bearings_in_range(sim):
    b = sim.measure_bearings()
    for i, row in b.items():
        for j, angle in row.items():
            assert -math.pi <= angle <= math.pi, (
                f"bearing b[{i}][{j}] = {angle:.4f} out of (-π, π]"
            )


def test_no_self_bearing(sim):
    b = sim.measure_bearings()
    for i in sim.robot_ids:
        assert i not in b[i]


def test_bearings_all_pairs_present(sim):
    b = sim.measure_bearings()
    for i in sim.robot_ids:
        for j in sim.robot_ids:
            if i != j:
                assert j in b[i]


def test_bearings_close_to_truth():
    """With zero noise the bearing should equal the true geometric angle."""
    sim = RobotSwarmSimulator(num_robots=2, bearing_noise=0.0, seed=1)
    sim.true_positions = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0])}
    sim.true_heading = {0: 0.0, 1: 0.0}
    b = sim.measure_bearings()
    assert b[0][1] == pytest.approx(0.0, abs=1e-9)   # robot 1 is directly ahead
    # _wrap maps π to -π; both represent the same angle
    assert abs(abs(b[1][0]) - math.pi) == pytest.approx(0.0, abs=1e-9)


# ── apply_control ────────────────────────────────────────────────────────────

def test_apply_control_moves_robot():
    sim = RobotSwarmSimulator(num_robots=2, seed=5)
    sim.true_positions = {0: np.array([0.0, 0.0]), 1: np.array([5.0, 5.0])}
    sim.true_heading = {0: 0.0, 1: 0.0}
    before = sim.true_positions[0].copy()
    sim.apply_control({0: np.array([1.0, 0.0])}, speed_limit=1.0)
    assert not np.allclose(sim.true_positions[0], before)


def test_apply_control_respects_speed_limit():
    sim = RobotSwarmSimulator(num_robots=1, seed=5)
    sim.true_positions = {0: np.array([0.0, 0.0])}
    sim.true_heading = {0: 0.0}
    limit = 0.02
    before = sim.true_positions[0].copy()
    sim.apply_control({0: np.array([10.0, 10.0])}, speed_limit=limit)
    step = np.linalg.norm(sim.true_positions[0] - before)
    assert step == pytest.approx(limit, rel=1e-6)


def test_apply_control_skips_missing_robots():
    sim = RobotSwarmSimulator(num_robots=2, seed=5)
    sim.true_positions = {0: np.array([0.0, 0.0]), 1: np.array([5.0, 5.0])}
    sim.true_heading = {0: 0.0, 1: 0.0}
    pos1_before = sim.true_positions[1].copy()
    sim.apply_control({0: np.array([1.0, 0.0])}, speed_limit=1.0)
    assert np.allclose(sim.true_positions[1], pos1_before)
