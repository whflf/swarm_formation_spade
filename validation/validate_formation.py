"""
Synchronous validation of the hybrid LVP algorithm (no XMPP overhead).
Reproduces the same computations as SPSAFormationAgent in-memory.

Expected results on seed=7:
  dist_res after warmup  < 0.10 m
  form_err (tail)        < 0.10 m
  scale                 ≈ 1.00
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from random import random

import numpy as np
from scipy.optimize import linear_sum_assignment

from swarm.config import (
    ALPHA,
    ALPHA_DECAY,
    AREA_SIZE,
    BEARING_NOISE,
    BETA,
    BETA_DECAY,
    CONTROL_ITERS,
    DECAY_EVERY,
    DIMENSIONS,
    EMA_H0,
    EMA_REL,
    FORMATION_SIDE,
    GAMMA,
    GAMMA_LVP,
    N_GRADIENT_SAMPLES,
    NEIGHBORS,
    NOISE_SCALE,
    SEED,
    SPEED_LIMIT,
    SPSA_PER_TICK,
    WARMUP_ITERS,
    N,
    joystick,
)
from swarm.robot_swarm_simulator import NoiseType, RobotSwarmSimulator

_Da = 1.0 / np.sqrt(DIMENSIONS)


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _make_corners(side: float = FORMATION_SIDE) -> dict[int, np.ndarray]:
    return {
        0: np.array([ side / 2,  side / 2]),
        1: np.array([-side / 2,  side / 2]),
        2: np.array([-side / 2, -side / 2]),
        3: np.array([ side / 2, -side / 2]),
    }


# ── simulator & state ────────────────────────────────────────────────────────

sim = RobotSwarmSimulator(
    num_robots=N, noise_type=NoiseType.UNIFORM,
    noise_scale=NOISE_SCALE, area_size=AREA_SIZE,
    seed=SEED, bearing_noise=BEARING_NOISE,
)

np.random.seed(SEED)
theta: dict[int, dict[int, np.ndarray]] = {
    i: {j: np.random.uniform(0, AREA_SIZE, size=2) for j in range(N)}
    for i in range(N)
}
heading = {i: 0.0 for i in range(N)}
rel_sin: dict[int, dict[int, float]] = {i: {} for i in range(N)}
rel_cos: dict[int, dict[int, float]] = {i: {} for i in range(N)}


# ── SPSA helpers ─────────────────────────────────────────────────────────────

def _observation(
    i: int,
    x_eval: np.ndarray,
    target: int,
    dist_i: dict[int, float],
    present: list[int],
    recv_dist: dict[int, dict[int, float]],
) -> float:
    self_pos = theta[i][i]
    total = count = 0.0
    d_it = dist_i.get(target)
    if d_it and d_it > 0.0:
        C = 2.0 * (theta[i][target] - self_pos)
        Cn = float(np.dot(C, C))
        if Cn > 1e-10:
            D = (float(np.dot(theta[i][target], theta[i][target]))
                 - float(np.dot(self_pos, self_pos)) + d_it ** 2)
            r = float(np.dot(C, x_eval)) - D
            total += r * r / Cn
            count += 1
        for sensor in present:
            if sensor == target:
                continue
            d_st = recv_dist.get(sensor, {}).get(target)
            if not d_st or d_st <= 0.0:
                continue
            C = 2.0 * (theta[i][sensor] - self_pos)
            Cn = float(np.dot(C, C))
            if Cn < 1e-10:
                continue
            D = (float(np.dot(theta[i][sensor], theta[i][sensor]))
                 - float(np.dot(self_pos, self_pos)) + d_it ** 2 - d_st ** 2)
            r = float(np.dot(C, x_eval)) - D
            total += r * r / Cn
            count += 1
    return total / max(count, 1)


def _spsa_all(dist: dict, alpha_loc: float, beta_loc: float) -> None:
    snap = {i: {j: theta[i][j].copy() for j in range(N)} for i in range(N)}
    new = {i: {j: theta[i][j].copy() for j in range(N)} for i in range(N)}
    for i in range(N):
        present = NEIGHBORS[i]
        dist_i = dist[i]
        recv_dist = {nb: dist[nb] for nb in present}
        for target in [i] + present:
            grad = np.zeros(DIMENSIONS)
            for _ in range(N_GRADIENT_SAMPLES):
                signs = [1 if random() < 0.5 else -1 for _ in range(DIMENSIONS)]
                delta = np.array([s * _Da for s in signs])
                yp = _observation(i, theta[i][target] + beta_loc * delta,
                                  target, dist_i, present, recv_dist)
                ym = _observation(i, theta[i][target] - beta_loc * delta,
                                  target, dist_i, present, recv_dist)
                grad += delta * (yp - ym) / (2 * beta_loc)
            grad /= N_GRADIENT_SAMPLES
            consensus = np.zeros(DIMENSIONS)
            for nb in present:
                consensus += snap[nb][target] - theta[i][target]
            consensus *= GAMMA
            new[i][target] = theta[i][target] - alpha_loc * grad + alpha_loc * consensus
    for i in range(N):
        for j in range(N):
            theta[i][j] = new[i][j]


def _dist_res() -> float:
    res = []
    for i in range(N):
        for j in NEIGHBORS[i]:
            td = np.linalg.norm(sim.true_positions[i] - sim.true_positions[j])
            ed = np.linalg.norm(theta[i][j] - theta[i][i])
            res.append(abs(td - ed))
    return float(np.mean(res))


def _formation_error(corners_assigned: dict[int, np.ndarray]) -> float:
    Q = np.array([corners_assigned[i] for i in range(N)])
    total = count = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            td = np.linalg.norm(Q[i] - Q[j])
            ad = np.linalg.norm(sim.true_positions[i] - sim.true_positions[j])
            total += abs(td - ad)
            count += 1
    return total / count


# ── Phase 1: warmup ──────────────────────────────────────────────────────────

alpha, beta = ALPHA, BETA
print(f"Warmup ({WARMUP_ITERS} итераций)...")
for it in range(WARMUP_ITERS):
    d = sim.get_noisy_distances()
    _spsa_all(d, alpha, beta)
    if (it + 1) % DECAY_EVERY == 0:
        alpha *= ALPHA_DECAY
        beta *= BETA_DECAY
    if (it + 1) % 2000 == 0:
        print(f"  iter {it + 1:5d}: dist_res={_dist_res():.3f} м")

print(f"\nПосле warmup: dist_res={_dist_res():.3f} м")

# ── Hungarian assignment ─────────────────────────────────────────────────────

corners = _make_corners()
map_pos = np.array([theta[0][j] for j in range(N)])
Q_world = np.array([corners[k] for k in range(N)])
Q_map = Q_world - Q_world.mean(axis=0) + map_pos.mean(axis=0)
cost = np.array([
    [np.linalg.norm(map_pos[i] - Q_map[k]) for k in range(N)]
    for i in range(N)
])
_, col = linear_sum_assignment(cost)
assignment = {i: col[i] for i in range(N)}
corners_assigned = {i: corners[assignment[i]] for i in range(N)}
d_star = {
    i: {j: float(np.linalg.norm(corners[assignment[j]] - corners[assignment[i]]))
        for j in NEIGHBORS[i]}
    for i in range(N)
}
print(f"Назначение: {assignment}")

# initial heading calibration
b0 = sim.measure_bearings()
for i in range(N):
    sin_s = cos_s = 0.0
    for j, phi in b0[i].items():
        diff = theta[i][j] - theta[i][i]
        if np.linalg.norm(diff) < 1e-6:
            continue
        h_j = np.arctan2(diff[1], diff[0]) - phi
        sin_s += np.sin(h_j)
        cos_s += np.cos(h_j)
    if sin_s != 0.0 or cos_s != 0.0:
        heading[i] = np.arctan2(sin_s, cos_s)

# ── Phase 2: control ─────────────────────────────────────────────────────────

print(f"\nФаза управления ({CONTROL_ITERS} тиков, {SPSA_PER_TICK} SPSA/тик)...")
fe_hist = []

for tick in range(CONTROL_ITERS):
    for _ in range(SPSA_PER_TICK):
        d = sim.get_noisy_distances()
        _spsa_all(d, alpha, beta)

    b = sim.measure_bearings()

    # robot 0: heading from map
    sin_s = cos_s = 0.0
    for j, phi in b[0].items():
        diff = theta[0][j] - theta[0][0]
        if np.linalg.norm(diff) < 1e-6:
            continue
        h_j = np.arctan2(diff[1], diff[0]) - phi
        sin_s += np.sin(h_j)
        cos_s += np.cos(h_j)
    if sin_s != 0.0 or cos_s != 0.0:
        new_h = np.arctan2(sin_s, cos_s)
        s = (1 - EMA_H0) * np.sin(heading[0]) + EMA_H0 * np.sin(new_h)
        c = (1 - EMA_H0) * np.cos(heading[0]) + EMA_H0 * np.cos(new_h)
        heading[0] = np.arctan2(s, c)

    h0 = heading[0]

    # robots 1..N-1: heading relative to robot 0
    for i in range(1, N):
        b0i = b[0].get(i)
        bi0 = b[i].get(0)
        if b0i is None or bi0 is None:
            continue
        delta = _wrap(b0i - bi0 - np.pi)
        if i not in rel_sin[i]:
            rel_sin[i][i] = np.sin(delta)
            rel_cos[i][i] = np.cos(delta)
        else:
            rel_sin[i][i] = (1 - EMA_REL) * rel_sin[i][i] + EMA_REL * np.sin(delta)
            rel_cos[i][i] = (1 - EMA_REL) * rel_cos[i][i] + EMA_REL * np.cos(delta)
        heading[i] = _wrap(h0 + np.arctan2(rel_sin[i][i], rel_cos[i][i]))

    d_fresh = sim.get_noisy_distances()
    joy = np.array(joystick(tick), dtype=float)
    chassis: dict[int, np.ndarray] = {}
    for i in range(N):
        c_i = theta[i][i]
        u = np.zeros(2)
        for j in NEIGHBORS[i]:
            d_ij = d_fresh[i].get(j)
            if d_ij is None:
                continue
            direction = theta[i][j] - c_i
            nrm = np.linalg.norm(direction)
            if nrm < 1e-9:
                continue
            u += (d_ij - d_star[i][j]) * (direction / nrm)
        u *= GAMMA_LVP
        h = heading[i]
        R = np.array([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
        chassis[i] = R @ u + joy
    sim.apply_control(chassis, SPEED_LIMIT)

    fe = _formation_error(corners_assigned)
    fe_hist.append(fe)
    if tick % 50 == 0:
        print(f"  tick {tick:3d}: form_err={fe:.3f} м  dist_res={_dist_res():.3f} м")

# ── results ───────────────────────────────────────────────────────────────────

fe_arr = np.array(fe_hist)

P_mat = np.array([sim.true_positions[i] for i in range(N)])
P_mat = P_mat - P_mat.mean(axis=0)
Q_mat = np.array([corners_assigned[i] for i in range(N)])
Q_mat = Q_mat - Q_mat.mean(axis=0)
_, S, _ = np.linalg.svd(P_mat.T @ Q_mat)
scale = S.sum() / (P_mat * P_mat).sum()

print(f"\n{'=' * 50}")
print(f"ИТОГ (seed={SEED}):")
print(f"  form_err min  = {fe_arr.min():.3f} м")
print(f"  form_err tail = {np.mean(fe_arr[-30:]):.3f} м")
print(f"  dist_res      = {_dist_res():.3f} м")
print(f"  scale         = {scale:.3f}")
