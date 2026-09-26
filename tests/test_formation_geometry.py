"""Tests for make_formation_polygon geometry."""
import numpy as np
import pytest

from swarm.spsa_formation_agent import make_formation_polygon, make_formation_square


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
def test_side_length(n):
    side = 2.0
    _, corners = make_formation_polygon(n, side=side)
    for i in range(n):
        j = (i + 1) % n
        actual = np.linalg.norm(corners[j] - corners[i])
        assert actual == pytest.approx(side, rel=1e-9), (
            f"n={n}: side {i}-{j} = {actual:.6f}, expected {side}"
        )


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
def test_centred_at_origin(n):
    _, corners = make_formation_polygon(n, side=2.0)
    centroid = np.mean([corners[i] for i in range(n)], axis=0)
    assert centroid == pytest.approx([0.0, 0.0], abs=1e-9)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
def test_vertex_count(n):
    _, corners = make_formation_polygon(n, side=2.0)
    assert len(corners) == n


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
def test_equal_radii(n):
    _, corners = make_formation_polygon(n, side=2.0)
    radii = [np.linalg.norm(corners[i]) for i in range(n)]
    assert max(radii) == pytest.approx(min(radii), rel=1e-9)


def test_offsets_self_zero():
    formations, _ = make_formation_polygon(4, side=2.0)
    for i in range(4):
        assert formations[i][i] == pytest.approx([0.0, 0.0], abs=1e-9)


def test_offsets_antisymmetric():
    formations, _ = make_formation_polygon(4, side=2.0)
    for i in range(4):
        for j in range(4):
            assert formations[i][j] == pytest.approx(-formations[j][i], abs=1e-9)


def test_square_alias_matches_polygon():
    _, corners_sq = make_formation_square(side=2.0)
    _, corners_poly = make_formation_polygon(n=4, side=2.0)
    for i in range(4):
        assert corners_sq[i] == pytest.approx(corners_poly[i], abs=1e-9)
