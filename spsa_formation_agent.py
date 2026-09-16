"""SPSA consensus + hybrid LVP formation control on SPADE."""
import asyncio
import json
import random

import numpy as np
from scipy.optimize import linear_sum_assignment
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message
from spade.template import Template

from config import (
    AGENT_UP_PROBABILITY,
    AGENTS_CREDENTIALS,
    ALPHA,
    ALPHA_DECAY,
    BETA,
    BETA_DECAY,
    DECAY_EVERY,
    DIMENSIONS,
    EMA_H0,
    EMA_REL,
    FORMATION_SIDE,
    GAMMA,
    GAMMA_LVP,
    LINK_PROBABILITY,
    N_GRADIENT_SAMPLES,
    NEIGHBORS,
    PER_ITER_TIMEOUT,
    PERIOD,
    SPEED_LIMIT,
    SPSA_PER_TICK,
    WARMUP_ITERS,
    N,
    joystick,
)


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def make_formation_polygon(n: int | None = None, side: float = FORMATION_SIDE):
    """
    Regular n-gon with the given side length, centred at the origin.
    For n=4 produces the same square as the original make_formation_square.
    Returns (formations, corners) where corners maps index → 2-D position.
    """
    if n is None:
        n = N
    R = side / (2.0 * np.sin(np.pi / n))
    phase = np.pi / n + np.pi / 2
    corners = {
        i: np.array([R * np.cos(phase + 2 * np.pi * i / n),
                     R * np.sin(phase + 2 * np.pi * i / n)])
        for i in range(n)
    }
    formations = {i: {j: corners[j] - corners[i] for j in range(n)} for i in range(n)}
    return formations, corners


def make_formation_square(side: float = FORMATION_SIDE):
    """Backward-compatible alias for make_formation_polygon(n=N)."""
    return make_formation_polygon(n=N, side=side)


class SPSAFormationAgent(Agent):
    def __init__(
        self,
        jid,
        password,
        index: int,
        neighbors_idx: list[int],
        center_jid: str,
        init_estimates: dict,
        sim,
        start_event: asyncio.Event,
        table: list,
    ) -> None:
        super().__init__(jid, password)
        self.index = index
        self.neighbors_idx = neighbors_idx
        self.center_jid = center_jid
        self.sim = sim
        self.start_event = start_event
        self.table = table

        self.theta: dict[int, np.ndarray] = {
            j: np.array(init_estimates[j], dtype=float) for j in range(N)
        }

        self.alpha = ALPHA
        self.beta = BETA
        self.gamma = GAMMA
        self.Delta_abs = 1.0 / np.sqrt(DIMENSIONS)

        self.recv_store: dict = {}
        self.recv_dist: dict = {}
        self.iteration = 0
        self.dist_store: dict[int, float] = {}

        self.msgs_sent = 0
        self.arith_ops = 0

        self.heading_est = 0.0
        self._rel_sin: dict[int, float] = {}
        self._rel_cos: dict[int, float] = {}

        self.assignment: dict[int, int] | None = None
        self.formation: dict[int, np.ndarray] | None = None
        self.d_star: dict[int, float] | None = None

        self.warmup_done = False
        self.control_tick = 0

    # ── heading estimation ───────────────────────────────────────────────────

    def _estimate_heading_from_map(self, bearings: dict[int, float]) -> float:
        sin_s = cos_s = 0.0
        for j, phi in bearings.items():
            diff = self.theta[j] - self.theta[self.index]
            if np.linalg.norm(diff) < 1e-6:
                continue
            h_j = np.arctan2(diff[1], diff[0]) - phi
            sin_s += np.sin(h_j)
            cos_s += np.cos(h_j)
        if sin_s == 0.0 and cos_s == 0.0:
            return self.heading_est
        return np.arctan2(sin_s, cos_s)

    def update_heading_robot0(self, bearings: dict[int, float]) -> None:
        new_h = self._estimate_heading_from_map(bearings)
        s = (1 - EMA_H0) * np.sin(self.heading_est) + EMA_H0 * np.sin(new_h)
        c = (1 - EMA_H0) * np.cos(self.heading_est) + EMA_H0 * np.cos(new_h)
        self.heading_est = np.arctan2(s, c)

    def update_heading_from_ref(
        self,
        h0_ref: float,
        bearing_0_to_me: float,
        my_bearing_to_0: float,
    ) -> None:
        i = self.index
        delta = _wrap(bearing_0_to_me - my_bearing_to_0 - np.pi)
        if i not in self._rel_sin:
            self._rel_sin[i] = np.sin(delta)
            self._rel_cos[i] = np.cos(delta)
        else:
            self._rel_sin[i] = (1 - EMA_REL) * self._rel_sin[i] + EMA_REL * np.sin(delta)
            self._rel_cos[i] = (1 - EMA_REL) * self._rel_cos[i] + EMA_REL * np.cos(delta)
        self.heading_est = _wrap(h0_ref + np.arctan2(self._rel_sin[i], self._rel_cos[i]))

    # ── chirality correction ─────────────────────────────────────────────────

    def detect_reflection(self, bearings: dict[int, float]) -> bool:
        """
        Return True when the local SPSA map has wrong chirality.
        Distance-only consensus cannot distinguish a configuration from its
        mirror; bearings disambiguate via the sign of cross-products.
        """
        self_pos = self.theta[self.index]
        nbs = [j for j in bearings if j != self.index]
        votes = total = 0
        for a in range(len(nbs)):
            for b in range(a + 1, len(nbs)):
                j, k = nbs[a], nbs[b]
                vj = self.theta[j] - self_pos
                vk = self.theta[k] - self_pos
                cross_map = vj[0] * vk[1] - vj[1] * vk[0]
                cross_real = np.sin(_wrap(bearings[k] - bearings[j]))
                if abs(cross_map) < 1e-9 or abs(cross_real) < 1e-6:
                    continue
                if np.sign(cross_map) != np.sign(cross_real):
                    votes += 1
                total += 1
        return (votes > total / 2) if total > 0 else False

    def fix_reflection(self) -> None:
        """Flip the map across the y-axis relative to its centroid."""
        center = np.array([self.theta[j] for j in range(N)]).mean(axis=0)
        for j in range(N):
            self.theta[j] = np.array([self.theta[j][0], 2 * center[1] - self.theta[j][1]])

    # ── Hungarian assignment ─────────────────────────────────────────────────

    def do_assignment(self, bearings: dict[int, float] | None = None) -> None:
        """Assign robots to formation corners (called once at end of warmup)."""
        if bearings is not None and self.detect_reflection(bearings):
            self.fix_reflection()
        _, corners = make_formation_square()
        map_pos = np.array([self.theta[j] for j in range(N)])
        map_center = map_pos.mean(axis=0)
        Q = np.array([corners[k] for k in range(N)])
        Q_map = Q - Q.mean(axis=0) + map_center
        cost = np.array([
            [np.linalg.norm(map_pos[i] - Q_map[k]) for k in range(N)]
            for i in range(N)
        ])
        _, col = linear_sum_assignment(cost)
        self.assignment = {i: col[i] for i in range(N)}
        my_corner = self.assignment[self.index]
        self.formation = {
            j: corners[self.assignment[j]] - corners[my_corner] for j in range(N)
        }
        self.d_star = {
            j: float(np.linalg.norm(self.formation[j]))
            for j in NEIGHBORS[self.index]
        }

    # ── hybrid LVP controller ────────────────────────────────────────────────

    def hybrid_lvp(self, dmeas: dict[int, float], joy: list[float]) -> np.ndarray:
        """
        u = γ · Σ_j (d_meas_ij − d*_ij) · n_ij   (in body frame)

        Direction n_ij from the slowly-evolving SPSA map preserves chirality.
        Magnitude (d_meas − d*) from a fresh range reading avoids map drift.
        """
        c_i = self.theta[self.index]
        u = np.zeros(2)
        for j in NEIGHBORS[self.index]:
            d_ij = dmeas.get(j)
            if d_ij is None:
                continue
            direction = self.theta[j] - c_i
            nrm = np.linalg.norm(direction)
            if nrm < 1e-9:
                continue
            u += (d_ij - self.d_star[j]) * (direction / nrm)
        u *= GAMMA_LVP
        h = self.heading_est
        R = np.array([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
        return R @ u + np.array(joy, dtype=float)

    # ── behaviours ───────────────────────────────────────────────────────────

    class SendAndUpdateBehaviour(PeriodicBehaviour):
        async def on_start(self) -> None:
            ag: "SPSAFormationAgent" = self.agent
            if ag.start_event is not None:
                await ag.start_event.wait()
            await asyncio.sleep(0.1)

        def _get_topology(self, it: int) -> tuple[bool, list[int]]:
            ag: "SPSAFormationAgent" = self.agent
            random.seed(f"agent-{ag.index}-{it}")
            is_alive = random.random() < AGENT_UP_PROBABILITY
            active_neighbors: list[int] = []
            if is_alive:
                for nb in ag.neighbors_idx:
                    edge_id = f"{min(ag.index, nb)}-{max(ag.index, nb)}-{it}"
                    random.seed(edge_id)
                    link_up = random.random() < LINK_PROBABILITY
                    random.seed(f"agent-{nb}-{it}")
                    nb_alive = random.random() < AGENT_UP_PROBABILITY
                    if link_up and nb_alive:
                        active_neighbors.append(nb)
            return is_alive, active_neighbors

        async def run(self) -> None:
            ag: "SPSAFormationAgent" = self.agent
            cur = ag.iteration
            is_alive, active_nbs = self._get_topology(cur)

            if not is_alive:
                while cur >= len(ag.table):
                    ag.table.append([None] * N)
                ag.table[cur][ag.index] = ag.theta.copy()
                ag.iteration += 1
                return

            noisy = ag.sim.get_noisy_distances()
            ag.dist_store = {
                nb: noisy[ag.index][nb]
                for nb in active_nbs
                if nb in noisy.get(ag.index, {})
            }

            payload_theta = {str(j): ag.theta[j].tolist() for j in range(N)}
            payload_dist = {str(k): float(v) for k, v in ag.dist_store.items()}
            for nb in active_nbs:
                msg = Message(to=AGENTS_CREDENTIALS[nb][0])
                msg.set_metadata("performative", "inform")
                msg.body = json.dumps({
                    "from_idx": ag.index,
                    "theta": payload_theta,
                    "dist": payload_dist,
                    "iter": cur,
                })
                await self.send(msg)
                ag.msgs_sent += 1

            waited = 0.0
            while not all(nb in ag.recv_store.get(cur, {}) for nb in active_nbs):
                await asyncio.sleep(0.01)
                waited += 0.01
                if waited >= PER_ITER_TIMEOUT:
                    break

            neighbors_theta = ag.recv_store.get(cur, {})
            neighbors_dist = ag.recv_dist.get(cur, {})
            present = [nb for nb in active_nbs if nb in neighbors_theta]

            ag.theta = self._spsa_step(present, neighbors_theta, neighbors_dist)

            if cur == WARMUP_ITERS - 1 and not ag.warmup_done:
                ag.warmup_done = True
                b = ag.sim.measure_bearings()
                ag.do_assignment(bearings=b[ag.index])
                ag.heading_est = ag._estimate_heading_from_map(b[ag.index])

            if ag.warmup_done and cur >= WARMUP_ITERS:
                if (cur - WARMUP_ITERS) % SPSA_PER_TICK == 0:
                    b = ag.sim.measure_bearings()
                    if ag.index == 0:
                        ag.update_heading_robot0(b[ag.index])
                    else:
                        b0_to_me = b[0].get(ag.index)
                        my_b_to_0 = b[ag.index].get(0)
                        if b0_to_me is not None and my_b_to_0 is not None:
                            h0_approx = ag._estimate_heading_from_map(b[0])
                            ag.update_heading_from_ref(h0_approx, b0_to_me, my_b_to_0)
                    joy = joystick(ag.control_tick)
                    u_chassis = ag.hybrid_lvp(ag.dist_store, joy)
                    ag.sim.apply_control({ag.index: u_chassis}, SPEED_LIMIT)
                    ag.control_tick += 1

            ag.iteration += 1
            while ag.iteration >= len(ag.table):
                ag.table.append([None] * N)
            ag.table[ag.iteration][ag.index] = ag.theta.copy()

            if cur > 0 and cur % DECAY_EVERY == 0:
                ag.alpha *= ALPHA_DECAY
                ag.beta *= BETA_DECAY

            ag.recv_store.pop(cur, None)
            ag.recv_dist.pop(cur, None)

        def _observation(
            self,
            x_eval: np.ndarray,
            target: int,
            present: list[int],
            neighbors_dist: dict,
        ) -> float:
            ag: "SPSAFormationAgent" = self.agent
            self_pos = ag.theta[ag.index]
            total = count = 0.0
            d_it = ag.dist_store.get(target)
            if d_it and d_it > 0.0:
                C = 2.0 * (ag.theta[target] - self_pos)
                Cn = float(np.dot(C, C))
                if Cn > 1e-10:
                    D = (float(np.dot(ag.theta[target], ag.theta[target]))
                         - float(np.dot(self_pos, self_pos)) + d_it ** 2)
                    r = float(np.dot(C, x_eval)) - D
                    total += r * r / Cn
                    count += 1
                for sensor in present:
                    if sensor == target:
                        continue
                    d_st = neighbors_dist.get(sensor, {}).get(target)
                    if not d_st or d_st <= 0.0:
                        continue
                    C = 2.0 * (ag.theta[sensor] - self_pos)
                    Cn = float(np.dot(C, C))
                    if Cn < 1e-10:
                        continue
                    D = (float(np.dot(ag.theta[sensor], ag.theta[sensor]))
                         - float(np.dot(self_pos, self_pos)) + d_it ** 2 - d_st ** 2)
                    r = float(np.dot(C, x_eval)) - D
                    total += r * r / Cn
                    count += 1
            return total / max(count, 1)

        def _spsa_step(
            self,
            present: list[int],
            neighbors_theta: dict,
            neighbors_dist: dict,
        ) -> dict[int, np.ndarray]:
            ag: "SPSAFormationAgent" = self.agent
            new_theta = {j: ag.theta[j].copy() for j in range(N)}
            for target in [ag.index] + present:
                grad = np.zeros(DIMENSIONS)
                for _ in range(N_GRADIENT_SAMPLES):
                    signs = [1 if random.random() < 0.5 else -1 for _ in range(DIMENSIONS)]
                    delta = np.array([s * ag.Delta_abs for s in signs])
                    yp = self._observation(
                        ag.theta[target] + ag.beta * delta, target, present, neighbors_dist
                    )
                    ym = self._observation(
                        ag.theta[target] - ag.beta * delta, target, present, neighbors_dist
                    )
                    grad += delta * (yp - ym) / (2 * ag.beta)
                grad /= N_GRADIENT_SAMPLES
                consensus = np.zeros(DIMENSIONS)
                for nb in present:
                    nb_est = np.array(neighbors_theta[nb][str(target)], dtype=float)
                    consensus += nb_est - ag.theta[target]
                consensus *= ag.gamma
                new_theta[target] = ag.theta[target] - ag.alpha * grad + ag.alpha * consensus
            return new_theta

    class ReceiverBehaviour(CyclicBehaviour):
        async def run(self) -> None:
            msg = await self.receive(timeout=0.1)
            if not msg:
                return
            try:
                data = json.loads(msg.body)
                frm = int(data["from_idx"])
                it = int(data.get("iter", 0))
                theta = {str(k): v for k, v in data["theta"].items()}
                dist = {int(k): float(v) for k, v in data.get("dist", {}).items()}
                self.agent.recv_store.setdefault(it, {})[frm] = theta
                self.agent.recv_dist.setdefault(it, {})[frm] = dist
            except Exception:
                pass

    async def setup(self) -> None:
        tmpl = Template()
        tmpl.set_metadata("performative", "inform")
        self.add_behaviour(self.ReceiverBehaviour(), tmpl)
        self.add_behaviour(self.SendAndUpdateBehaviour(period=PERIOD))
