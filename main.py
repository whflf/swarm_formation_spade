"""
main.py — запуск SPSA-консенсуса + гибридного LVP на SPADE.

Запуск:
    python3 main.py

Фазы:
  1. Warmup (WARMUP_ITERS итераций): чистый SPSA, рой стоит.
  2. Управление (CONTROL_ITERS тиков × SPSA_PER_TICK итераций):
     гибридный LVP + движение под джойстиком.
"""
import sys

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import asyncio
import numpy as np
import spade

from spsa_formation_agent import SPSAFormationAgent, make_formation_square
from robot_swarm_simulator import RobotSwarmSimulator, NoiseType
from config import *


def dist_residual(table_row, sim):
    res = []
    for i in range(N):
        if table_row[i] is None:
            continue
        theta_i = table_row[i]
        for nb in NEIGHBORS[i]:
            true_d = np.linalg.norm(sim.true_positions[i] - sim.true_positions[nb])
            est_d = np.linalg.norm(theta_i[nb] - theta_i[i])
            res.append(abs(true_d - est_d))
    return float(np.mean(res)) if res else 0.0


def formation_error(sim, corners_assigned):
    Q = np.array([corners_assigned[i] for i in range(N)])
    v, c = 0.0, 0
    for i in range(N):
        for j in range(i + 1, N):
            td = np.linalg.norm(Q[i] - Q[j])
            ad = np.linalg.norm(sim.true_positions[i] - sim.true_positions[j])
            v += abs(td - ad)
            c += 1
    return v / c if c else 0.0


async def main():
    sim = RobotSwarmSimulator(
        num_robots=N,
        noise_type=NoiseType.UNIFORM,
        noise_scale=NOISE_SCALE,
        area_size=AREA_SIZE,
        seed=SEED,
        bearing_noise=BEARING_NOISE,
    )

    np.random.seed(SEED)
    init_estimates = {
        i: {j: np.random.uniform(0, AREA_SIZE, size=2) for j in range(N)}
        for i in range(N)
    }

    iterations_table = [[None] * N]
    iterations_table[0] = [
        {j: init_estimates[i][j].copy() for j in range(N)} for i in range(N)
    ]

    start_event = asyncio.Event()
    agents = []
    for i in range(N):
        jid, pwd = AGENTS_CREDENTIALS[i]
        a = SPSAFormationAgent(
            jid, pwd, i, NEIGHBORS[i], CENTER_JID,
            init_estimates={j: init_estimates[i][j] for j in range(N)},
            sim=sim,
            start_event=start_event,
            table=iterations_table,
        )
        agents.append(a)

    await asyncio.gather(*(a.start(auto_register=True) for a in agents))
    start_event.set()
    await asyncio.sleep(0.5)

    print("=" * 60)
    print(f"Рой: {N} роботов, квадрат {FORMATION_SIDE}x{FORMATION_SIDE} м, seed={SEED}")
    print(f"Warmup: {WARMUP_ITERS} iter  |  Управление: {CONTROL_ITERS} тиков")
    print("=" * 60)

    t0 = asyncio.get_event_loop().time()
    last_reported = -1
    prev_warmup_done = False
    corners_assigned = {}

    while True:
        cur = min(a.iteration for a in agents)

        if cur <= WARMUP_ITERS and cur > last_reported and cur % 1000 == 0:
            row = iterations_table[cur] if cur < len(iterations_table) else [None] * N
            dr = dist_residual(row, sim)
            print(f"  [warmup] iter={cur:5d}  dist_res={dr:.3f} м")
            last_reported = cur

        warmup_done_now = all(a.warmup_done for a in agents)
        if warmup_done_now and not prev_warmup_done:
            prev_warmup_done = True
            _, corners = make_formation_square()
            corners_assigned = {i: corners[agents[0].assignment[i]] for i in range(N)}
            print(f"\nWarmup завершён. Назначение (агент 0): {agents[0].assignment}")
            print(f"\n{'тик':>5} | {'form_err':>10} | {'dist_res':>10}")
            print("-" * 32)
            last_reported = -1

        if warmup_done_now and corners_assigned:
            ticks_done = min(a.control_tick for a in agents)
            if ticks_done > last_reported and ticks_done % 50 == 0:
                row = iterations_table[-1] if iterations_table else [None] * N
                dr = dist_residual(row, sim)
                fe = formation_error(sim, corners_assigned)
                print(f"  {ticks_done:3d}  | {fe:10.3f} м | {dr:10.3f} м")
                last_reported = ticks_done

        if cur >= ITERATIONS:
            break
        if asyncio.get_event_loop().time() - t0 > 1800:
            print("  [timeout]")
            break
        await asyncio.sleep(0.2)

    await asyncio.gather(*(a.stop() for a in agents))

    fe_final = formation_error(sim, corners_assigned) if corners_assigned else float('nan')
    row_final = iterations_table[-1] if iterations_table else [None] * N
    dr_final = dist_residual(row_final, sim)

    print("\n" + "=" * 60)
    print(f"ИТОГ:  form_err={fe_final:.3f} м   dist_res={dr_final:.3f} м")
    print("\nИстинные позиции:")
    for i in range(N):
        print(f"  р{i}: {sim.true_positions[i].round(3)}")


if __name__ == "__main__":
    spade.run(main(), embedded_xmpp_server=True)
