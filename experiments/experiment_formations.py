"""
Test the full SPSA + LVP pipeline for arbitrary formation shapes.
Includes chirality correction.

Usage:
    python experiment_formations.py --n 6 --seed 7 --formation polygon
    python experiment_formations.py --n 8 --formation grid
"""
import argparse

import numpy as np
from scipy.optimize import linear_sum_assignment

from swarm import config as C
from swarm.robot_swarm_simulator import NoiseType, RobotSwarmSimulator

_Da = 1.0 / np.sqrt(2)


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── formation generators ──────────────────────────────────────────────────────

def formation_line(n: int, spacing: float = 1.5) -> dict[int, np.ndarray]:
    """Linear array along the x-axis."""
    xs = (np.arange(n) - (n - 1) / 2) * spacing
    return {i: np.array([xs[i], 0.0]) for i in range(n)}


def formation_grid(n: int, spacing: float = 1.5) -> dict[int, np.ndarray]:
    """Rectangular grid, roughly square, centred at origin."""
    cols = int(np.ceil(np.sqrt(n)))
    pts = {i: np.array([i % cols * spacing, i // cols * spacing], dtype=float)
           for i in range(n)}
    ctr = np.mean(list(pts.values()), axis=0)
    return {i: pts[i] - ctr for i in range(n)}


def formation_vshape(n: int, spacing: float = 1.2) -> dict[int, np.ndarray]:
    """V-formation (leader at front, two wings spreading back)."""
    pts = {0: np.array([0.0, 0.0])}
    for i in range(1, n):
        side = 1 if i % 2 == 1 else -1
        rank = (i + 1) // 2
        pts[i] = np.array([side * rank * spacing, -rank * spacing])
    ctr = np.mean(list(pts.values()), axis=0)
    return {i: pts[i] - ctr for i in range(n)}


def formation_polygon(n: int, side: float = 2.0) -> dict[int, np.ndarray]:
    """Regular n-gon with the given side length."""
    R = side / (2.0 * np.sin(np.pi / n))
    ph = np.pi / n + np.pi / 2
    return {
        i: np.array([R * np.cos(ph + 2 * np.pi * i / n),
                     R * np.sin(ph + 2 * np.pi * i / n)])
        for i in range(n)
    }


FORMATIONS: dict[str, object] = {
    "polygon": formation_polygon,
    "line": formation_line,
    "grid": formation_grid,
    "vshape": formation_vshape,
}


# ── chirality detection ───────────────────────────────────────────────────────

def _detect_reflection(
    theta: np.ndarray,
    bearings: dict[int, float],
    idx: int,
) -> bool:
    sp = theta[idx]
    nbs = [j for j in bearings if j != idx]
    votes = total = 0
    for a in range(len(nbs)):
        for b in range(a + 1, len(nbs)):
            j, k = nbs[a], nbs[b]
            vj = theta[j] - sp
            vk = theta[k] - sp
            cross_map = vj[0] * vk[1] - vj[1] * vk[0]
            cross_real = np.sin(_wrap(bearings[k] - bearings[j]))
            if abs(cross_map) < 1e-9 or abs(cross_real) < 1e-6:
                continue
            if np.sign(cross_map) != np.sign(cross_real):
                votes += 1
            total += 1
    return (votes > total / 2) if total > 0 else False


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(
    n: int,
    seed: int,
    formation_fn,
    warmup: int = 4000,
    ctrl: int = 120,
    spt: int = 60,
    fix_chirality: bool = True,
) -> dict:
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

    def _obs(i: int, xe: np.ndarray, tgt: int, Mrow: np.ndarray, Mall: np.ndarray) -> np.ndarray:
        sp = theta[i, i]
        d_it = Mrow[tgt]
        acc = np.zeros(len(xe))
        cnt = 0
        if d_it > 0:
            Cv = 2 * (theta[i, tgt] - sp)
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
                Cv = 2 * (theta[i, s] - sp)
                Cn = float(Cv @ Cv)
                if Cn < 1e-10:
                    continue
                D = float(theta[i, s] @ theta[i, s] - sp @ sp + d_it ** 2 - d_st ** 2)
                acc += (xe @ Cv - D) ** 2 / Cn
                cnt += 1
        return acc / max(cnt, 1)

    def _spsa(a: float, b: float) -> np.ndarray:
        M = _noisy_matrix()
        new = theta.copy()
        for i in range(n):
            for tgt in range(n):
                dl = np.random.choice([-1, 1], size=(C.N_GRADIENT_SAMPLES, 2)) * _Da
                base = theta[i, tgt]
                yp = _obs(i, base[None, :] + b * dl, tgt, M[i], M)
                ym = _obs(i, base[None, :] - b * dl, tgt, M[i], M)
                grad = (((yp - ym) / (2 * b))[:, None] * dl).mean(0)
                cons = C.GAMMA * np.sum(theta[:, tgt] - theta[i, tgt], axis=0)
                new[i, tgt] = theta[i, tgt] - a * grad + a * cons
        return new

    a, b = C.ALPHA, C.BETA
    for it in range(warmup):
        theta = _spsa(a, b)
        if (it + 1) % C.DECAY_EVERY == 0:
            a *= C.ALPHA_DECAY
            b *= C.BETA_DECAY

    b_init = sim.measure_bearings()
    reflected = _detect_reflection(theta[0], b_init[0], 0)
    if fix_chirality and reflected:
        for i in range(n):
            ctr = theta[i].mean(axis=0)
            theta[i, :, 1] = 2 * ctr[1] - theta[i, :, 1]

    corners = formation_fn(n)
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

    heading = np.zeros(n)
    rel_s = np.zeros(n)
    rel_c = np.zeros(n)
    for i in range(n):
        sin_s = cos_s = 0.0
        for j, phi in b_init[i].items():
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

    fes = []
    for tick in range(ctrl):
        for _ in range(spt):
            theta = _spsa(a, b)

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
            u *= C.GAMMA_LVP
            h = heading[i]
            R = np.array([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
            ch[i] = R @ u + joy
        sim.apply_control(ch, C.SPEED_LIMIT)
        fes.append(_form_err())

    fes_arr = np.array(fes)
    return {
        "reflected": reflected,
        "fe_min": float(fes_arr.min()),
        "fe_tail": float(fes_arr[-30:].mean()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test swarm formation shapes")
    parser.add_argument("--n", type=int, default=4, help="Number of robots")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--formation", choices=list(FORMATIONS), default="polygon",
        help="Formation shape",
    )
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--ctrl", type=int, default=120)
    parser.add_argument("--spt", type=int, default=60, help="SPSA steps per tick")
    args = parser.parse_args()

    fn = FORMATIONS[args.formation]
    result = run(args.n, args.seed, fn,
                 warmup=args.warmup, ctrl=args.ctrl, spt=args.spt)
    print(
        f"N={args.n} {args.formation} seed={args.seed}: "
        f"reflected={result['reflected']} "
        f"fe_min={result['fe_min']:.3f} "
        f"fe_tail={result['fe_tail']:.3f}"
    )
