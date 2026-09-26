"""Tests for chirality (reflection) detection."""
import math
import numpy as np

from swarm.spsa_formation_agent import detect_map_reflection, make_formation_polygon


def _bearings_from_positions(
    positions: dict[int, np.ndarray],
    observer: int,
) -> dict[int, float]:
    """Exact (noiseless) bearings assuming heading=0."""
    return {
        j: math.atan2((pos - positions[observer])[1], (pos - positions[observer])[0])
        for j, pos in positions.items()
        if j != observer
    }


# ── correct map ──────────────────────────────────────────────────────────────

def test_no_reflection_square():
    positions = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 1.0]),
        3: np.array([0.0, 1.0]),
    }
    bearings = _bearings_from_positions(positions, observer=0)
    assert not detect_map_reflection(positions, bearings, self_idx=0)


def test_no_reflection_triangle():
    positions = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 0.0]),
        2: np.array([0.5, math.sqrt(3) / 2]),
    }
    bearings = _bearings_from_positions(positions, observer=0)
    assert not detect_map_reflection(positions, bearings, self_idx=0)


# ── mirrored map ─────────────────────────────────────────────────────────────

def test_reflection_detected_square():
    real = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 1.0]),
        3: np.array([0.0, 1.0]),
    }
    mirrored = {k: np.array([v[0], -v[1]]) for k, v in real.items()}
    bearings = _bearings_from_positions(real, observer=0)
    assert detect_map_reflection(mirrored, bearings, self_idx=0)


def test_reflection_detected_triangle():
    real = {
        0: np.array([0.0, 0.0]),
        1: np.array([2.0, 0.0]),
        2: np.array([1.0, 1.5]),
    }
    mirrored = {k: np.array([v[0], -v[1]]) for k, v in real.items()}
    bearings = _bearings_from_positions(real, observer=0)
    assert detect_map_reflection(mirrored, bearings, self_idx=0)


# ── edge cases ───────────────────────────────────────────────────────────────

def test_only_one_neighbour_returns_false():
    # Single neighbour → no pairs to vote → must not crash, must return False
    positions = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0])}
    bearings = _bearings_from_positions(positions, observer=0)
    assert not detect_map_reflection(positions, bearings, self_idx=0)


def test_fix_reflection_restores_chirality():
    _, corners = make_formation_polygon(n=4, side=2.0)
    real = dict(corners)
    mirrored = {k: np.array([v[0], -v[1]]) for k, v in real.items()}
    bearings = _bearings_from_positions(real, observer=0)

    assert detect_map_reflection(mirrored, bearings, self_idx=0)

    center = np.mean(list(mirrored.values()), axis=0)
    fixed = {k: np.array([v[0], 2 * center[1] - v[1]]) for k, v in mirrored.items()}

    assert not detect_map_reflection(fixed, bearings, self_idx=0)
