"""
Vectorised full pipeline (warmup → assignment → control) for large N.
Uses numpy array operations instead of dicts; suitable for scaling studies.

Usage:
    python experiment_scale_fast.py --n 20 --seed 7 --warmup 8000 --ctrl 150
    python experiment_scale_fast.py --n 4  --seed 7 --warmup 200  --ctrl 5  # quick test
"""
import argparse
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from swarm import config as C
from swarm.robot_swarm_simulator import NoiseType, RobotSwarmSimulator

_Da = 1.0 / np.sqrt(2)


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _make_polygon(n: int, side: float = 2.0) -> dict[int, np.ndarray]:
    R = side / (2.0 * np.sin(np.pi / n))
    ph = np.pi / n + np.pi / 2
    return {
        i: np.array([R * np.cos(ph + 2 * np.pi * i / n),
                     R * np.sin(ph + 2 * np.pi * i / n)])
        for i in range(n)
    }


def run(
    n: int,
    seed: int,
    warmup: int = 8000,
    ctrl: int = 150,
    spt: int = 80,
    gamma_lvp: float | None = None,
    speed_limit: float | None = None,
    verbose: bool = False,
) -> dict:
    gamma_lvp = gamma_lvp if gamma_lvp is not None else C.GAMMA_LVP
    speed_limit = speed_limit if speed_limit is not None else C.SPEED_LIMIT

    sim = RobotSwarmSimulator(
        num_robots=n, noise_type=NoiseType.UNIFORM,
        noise_scale=C.NOISE_SCALE, area_size=C.AREA_SIZE,
        seed=seed, bearing_noise=C.BEARING_NOISE,
    )
    np.random.seed(seed)
    theta = np.random.uniform(0, C.AREA_SIZE, size=(n, n, 2))

    nei = [[j for j in range(n) if j != i] for i in range(n)]

    def _noisy_matrix() -> np.ndarray:
        d = sim.get_noisy_distances()
        M = np.zeros((n, n))
        for i in range(n):
            for j, v in d[i].items():
                M[i, j] = v
        return M

    def _obs_batch(
        i: int, xe: np.ndarray, tgt: int, Mrow: np.ndarray, Mall: np.ndarray
    ) -> np.ndarray:
        sp = theta[i, i]
        d_it = Mrow[tgt]
        acc = np.zeros(len(xe))
        cnt = 0
        if d_it > 0:
            Cv = 2.0 * (theta[i, tgt] - sp)
            Cn = float(Cv @ Cv)
            if Cn > 1e-10:
                D = float(theta[i, tgt] @ theta[i, tgt] - sp @ sp + d_it ** 2)
                acc += (xe @ Cv - D) ** 2 / Cn
                cnt += 1
            for s in range(n):
                if s == i or s == tgt:
                    continue
                d_st = Mall[s, tgt]
                if d_st <= 0:
                    continue
                Cv = 2.0 * (theta[i, s] - sp)
                Cn = float(Cv @ Cv)
                if Cn < 1e-10:
                    continue
                D = float(theta[i, s] @ theta[i, s] - sp @ sp + d_it ** 2 - d_st ** 2)
                acc += (xe @ Cv - D) ** 2 / Cn
                cnt += 1
        return acc / max(cnt, 1)

    def _spsa_step(a: float, b: float) -> np.ndarray:
        M = _noisy_matrix()
        new = theta.copy()
        for i in range(n):
            for tgt in range(n):
                signs = np.random.choice([-1, 1], size=(C.N_GRADIENT_SAMPLES, 2))
                dl = signs * _Da
                base = theta[i, tgt]
                yp = _obs_batch(i, base[None, :] + b * dl, tgt, M[i], M)
                ym = _obs_batch(i, base[None, :] - b * dl, tgt, M[i], M)
                grad = (((yp - ym) / (2 * b))[:, None] * dl).mean(0)
                cons = C.GAMMA * np.sum(theta[:, tgt] - theta[i, tgt], axis=0)
                new[i, tgt] = theta[i, tgt] - a * grad + a * cons
        return new

    def _dist_res() -> float:
        res = [
            abs(
                np.linalg.norm(sim.true_positions[i] - sim.true_positions[j])
                - np.linalg.norm(theta[i, j] - theta[i, i])
            )
            for i in range(n)
            for j in range(n)
            if i != j
        ]
        return float(np.mean(res))

    # warmup
    a, b = C.ALPHA, C.BETA
    for it in range(warmup):
        theta = _spsa_step(a, b)
        if (it + 1) % C.DECAY_EVERY == 0:
            a *= C.ALPHA_DECAY
            b *= C.BETA_DECAY
    dr_warmup = _dist_res()

    # Hungarian assignment
    corners = _make_polygon(n, side=C.FORMATION_SIDE)
    mp = theta[0]
    Qw = np.array([corners[k] for k in range(n)])
    Qm = Qw - Qw.mean(axis=0) + mp.mean(axis=0)
    cost = np.linalg.norm(mp[:, None, :] - Qm[None, :, :], axis=2)
    _, col = linear_sum_assignment(cost)
    asg = {i: col[i] for i in range(n)}
    corners_asg = {i: corners[asg[i]] for i in range(n)}
    d_star = {
        i: {j: float(np.linalg.norm(corners[asg[j]] - corners[asg[i]])) for j in nei[i]}
        for i in range(n)
    }

    # heading initialisation
    heading = np.zeros(n)
    rel_s = np.zeros(n)
    rel_c = np.zeros(n)
    b0 = sim.measure_bearings()
    for i in range(n):
        sin_s = cos_s = 0.0
        for j, phi in b0[i].items():
            diff = theta[i, j] - theta[i, i]
            if np.linalg.norm(diff) < 1e-6:
                continue
            hj = np.arctan2(diff[1], diff[0]) - phi
            sin_s += np.sin(hj)
            cos_s += np.cos(hj)
        if sin_s or cos_s:
            heading[i] = np.arctan2(sin_s, cos_s)

    def _form_err() -> float:
        Q = np.array([corners_asg[i] for i in range(n)])
        total = count = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                total += abs(
                    np.linalg.norm(Q[i] - Q[j])
                    - np.linalg.norm(sim.true_positions[i] - sim.true_positions[j])
                )
                count += 1
        return total / count

    # control phase
    fe_hist = []
    for tick in range(ctrl):
        for _ in range(spt):
            theta = _spsa_step(a, b)

        bb = sim.measure_bearings()
        sin_s = cos_s = 0.0
        for j, phi in bb[0].items():
            diff = theta[0, j] - theta[0, 0]
            if np.linalg.norm(diff) < 1e-6:
                continue
            hj = np.arctan2(diff[1], diff[0]) - phi
            sin_s += np.sin(hj)
            cos_s += np.cos(hj)
        if sin_s or cos_s:
            nh = np.arctan2(sin_s, cos_s)
            heading[0] = np.arctan2(
                (1 - C.EMA_H0) * np.sin(heading[0]) + C.EMA_H0 * np.sin(nh),
                (1 - C.EMA_H0) * np.cos(heading[0]) + C.EMA_H0 * np.cos(nh),
            )

        h0 = heading[0]
        for i in range(1, n):
            b0i = bb[0].get(i)
            bi0 = bb[i].get(0)
            if b0i is None or bi0 is None:
                continue
            dl = _wrap(b0i - bi0 - np.pi)
            rel_s[i] = (1 - C.EMA_REL) * rel_s[i] + C.EMA_REL * np.sin(dl)
            rel_c[i] = (1 - C.EMA_REL) * rel_c[i] + C.EMA_REL * np.cos(dl)
            heading[i] = _wrap(h0 + np.arctan2(rel_s[i], rel_c[i]))

        Mf = _noisy_matrix()
        joy = np.array(C.joystick(tick), dtype=float)
        ch: dict[int, np.ndarray] = {}
        for i in range(n):
            u = np.zeros(2)
            ci = theta[i, i]
            for j in nei[i]:
                dij = Mf[i, j]
                if dij <= 0:
                    continue
                direction = theta[i, j] - ci
                nrm = np.linalg.norm(direction)
                if nrm < 1e-9:
                    continue
                u += (dij - d_star[i][j]) * (direction / nrm)
            u *= gamma_lvp
            h = heading[i]
            R = np.array([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
            ch[i] = R @ u + joy
        sim.apply_control(ch, speed_limit)
        fe_hist.append(_form_err())
        if verbose and tick % 30 == 0:
            print(f"    tick {tick}: fe={fe_hist[-1]:.3f}")

    fe = np.array(fe_hist)
    P = np.array([sim.true_positions[i] for i in range(n)])
    P = P - P.mean(axis=0)
    Q = np.array([corners_asg[i] for i in range(n)])
    Q = Q - Q.mean(axis=0)
    _, S, _ = np.linalg.svd(P.T @ Q)
    scale = float(S.sum() / (P * P).sum())

    return {
        "n": n,
        "seed": seed,
        "dr_warmup": dr_warmup,
        "fe_min": float(fe.min()),
        "fe_tail": float(fe[-30:].mean()),
        "scale": scale,
        "dr_final": _dist_res(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast scaling test for the SPSA+LVP pipeline")
    parser.add_argument("--n", type=int, default=4, help="Number of robots")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=8000)
    parser.add_argument("--ctrl", type=int, default=150)
    parser.add_argument("--spt", type=int, default=80, help="SPSA steps per tick")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    r = run(args.n, args.seed, warmup=args.warmup, ctrl=args.ctrl,
            spt=args.spt, verbose=args.verbose)
    elapsed = time.time() - t0
    print(
        f"N={r['n']} seed={r['seed']} ({elapsed:.0f}s): "
        f"dr_wu={r['dr_warmup']:.3f} "
        f"fe_tail={r['fe_tail']:.3f} "
        f"scale={r['scale']:.3f}"
    )
