"""
spsa_formation_agent.py — SPSA-консенсус для построения карты + LVP для движения.
"""

import asyncio
import json
import random
import numpy as np

from scipy.optimize import linear_sum_assignment

from spade.agent import Agent
from spade.behaviour import PeriodicBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template

from config import *


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def make_formation_square(side=FORMATION_SIDE):
    """Квадрат: роботы 0-3 по углам. Возвращает {i: pos} и {i: {j: offset}}."""
    corners = {
        0: np.array([side / 2, side / 2]),
        1: np.array([-side / 2, side / 2]),
        2: np.array([-side / 2, -side / 2]),
        3: np.array([side / 2, -side / 2]),
    }
    formations = {i: {j: corners[j] - corners[i] for j in range(N)}
                  for i in range(N)}
    return formations, corners


class SPSAFormationAgent(Agent):
    def __init__(self, jid, password, index, neighbors_idx, center_jid,
                 init_estimates, sim, start_event, table):
        super().__init__(jid, password)
        self.index = index
        self.neighbors_idx = neighbors_idx
        self.center_jid = center_jid
        self.sim = sim
        self.start_event = start_event
        self.table = table

        # оценки позиций
        self.theta = {j: np.array(init_estimates[j], dtype=float)
                      for j in range(N)}

        # SPSA-параметры
        self.alpha = ALPHA
        self.beta = BETA
        self.gamma = GAMMA
        self.Delta_abs = 1.0 / np.sqrt(DIMENSIONS)

        # синхронизация
        self.recv_store = {}
        self.recv_dist = {}
        self.iteration = 0
        self.dist_store = {}

        # heading
        self.heading_est = 0.0
        self._rel_sin = {}
        self._rel_cos = {}

        # формация (заполняется после warmup)
        self.assignment = None  # {i: corner_idx}
        self.formation = None  # {j: offset_vector}
        self.d_star = None  # {j: desired_distance}

        # фаза
        self.warmup_done = False
        self.control_tick = 0

    # ------------------------------------------------------------------
    # Heading
    # ------------------------------------------------------------------
    def _estimate_heading_from_map(self, bearings):
        sin_s, cos_s = 0.0, 0.0
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

    def update_heading_robot0(self, bearings):
        new_h = self._estimate_heading_from_map(bearings)
        s = (1 - EMA_H0) * np.sin(self.heading_est) + EMA_H0 * np.sin(new_h)
        c = (1 - EMA_H0) * np.cos(self.heading_est) + EMA_H0 * np.cos(new_h)
        self.heading_est = np.arctan2(s, c)

    def update_heading_from_ref(self, h0_ref, bearing_0_to_me, my_bearing_to_0):
        i = self.index
        delta = _wrap(bearing_0_to_me - my_bearing_to_0 - np.pi)
        if i not in self._rel_sin:
            self._rel_sin[i] = np.sin(delta)
            self._rel_cos[i] = np.cos(delta)
        else:
            self._rel_sin[i] = (1 - EMA_REL) * self._rel_sin[i] + EMA_REL * np.sin(delta)
            self._rel_cos[i] = (1 - EMA_REL) * self._rel_cos[i] + EMA_REL * np.cos(delta)
        self.heading_est = _wrap(h0_ref + np.arctan2(self._rel_sin[i], self._rel_cos[i]))

    # ------------------------------------------------------------------
    # Hungarian assignment (вызывается один раз после warmup)
    # ------------------------------------------------------------------
    def do_assignment(self):
        _, corners = make_formation_square()
        map_pos = np.array([self.theta[j] for j in range(N)])
        map_center = map_pos.mean(0)
        Q = np.array([corners[k] for k in range(N)])
        Q_map = Q - Q.mean(0) + map_center
        cost = np.array([[np.linalg.norm(map_pos[i] - Q_map[k])
                          for k in range(N)] for i in range(N)])
        _, col = linear_sum_assignment(cost)
        self.assignment = {i: col[i] for i in range(N)}

        my_corner = self.assignment[self.index]
        self.formation = {
            j: corners[self.assignment[j]] - corners[my_corner]
            for j in range(N)
        }
        self.d_star = {j: float(np.linalg.norm(self.formation[j]))
                       for j in NEIGHBORS[self.index]}

    # ------------------------------------------------------------------
    # Гибридный LVP
    # ------------------------------------------------------------------
    def hybrid_lvp(self, dmeas, joystick):
        """
        u = γ · Σ_j (d_meas_ij − d*_ij) · n_ij   [в локальной карте]
        Возвращает вектор управления в СК корпуса.
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
        # перевод в СК корпуса
        h = self.heading_est
        R = np.array([[np.cos(h), np.sin(h)],
                      [-np.sin(h), np.cos(h)]])
        return R @ u + np.array(joystick, dtype=float)

    class SendAndUpdateBehaviour(PeriodicBehaviour):
        async def on_start(self):
            ag: "SPSAFormationAgent" = self.agent
            if ag.start_event is not None:
                await ag.start_event.wait()
            await asyncio.sleep(0.1)

        def get_topology_for_iter(self, it):
            ag: "SPSAFormationAgent" = self.agent
            random.seed(f"agent-{ag.index}-{it}")
            is_alive = random.random() < AGENT_UP_PROBABILITY
            active_neighbors = []
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

        async def run(self):
            ag: "SPSAFormationAgent" = self.agent
            cur_iter = ag.iteration
            is_alive, active_neighbors = self.get_topology_for_iter(cur_iter)

            if not is_alive:
                while cur_iter >= len(ag.table):
                    ag.table.append([None] * N)
                ag.table[cur_iter][ag.index] = ag.theta.copy()
                ag.iteration += 1
                return

            # измерение расстояний
            noisy = ag.sim.get_noisy_distances()
            ag.dist_store = {nb: noisy[ag.index][nb]
                             for nb in active_neighbors
                             if nb in noisy.get(ag.index, {})}

            # рассылка
            payload_theta = {str(j): ag.theta[j].tolist() for j in range(N)}
            payload_dist = {str(k): float(v) for k, v in ag.dist_store.items()}
            for nb in active_neighbors:
                msg = Message(to=AGENTS_CREDENTIALS[nb][0])
                msg.set_metadata("performative", "inform")
                msg.body = json.dumps({
                    "from_idx": ag.index,
                    "theta": payload_theta,
                    "dist": payload_dist,
                    "iter": cur_iter,
                })
                await self.send(msg)

            # ожидание соседей
            waited = 0.0
            while True:
                if all(nb in ag.recv_store.get(cur_iter, {})
                       for nb in active_neighbors):
                    break
                await asyncio.sleep(0.01)
                waited += 0.01
                if waited >= PER_ITER_TIMEOUT:
                    break

            neighbors_theta = ag.recv_store.get(cur_iter, {})
            neighbors_dist = ag.recv_dist.get(cur_iter, {})
            present = [nb for nb in active_neighbors if nb in neighbors_theta]

            # SPSA
            new_theta = self._spsa_step(present, neighbors_theta, neighbors_dist)
            ag.theta = new_theta

            if cur_iter == WARMUP_ITERS - 1 and not ag.warmup_done:
                ag.warmup_done = True
                ag.do_assignment()
                b = ag.sim.measure_bearings()
                ag.heading_est = ag._estimate_heading_from_map(b[ag.index])

            if ag.warmup_done and cur_iter >= WARMUP_ITERS:
                if (cur_iter - WARMUP_ITERS) % SPSA_PER_TICK == 0:
                    b = ag.sim.measure_bearings()
                    if ag.index == 0:
                        ag.update_heading_robot0(b[ag.index])
                    else:
                        b0_to_me = b[0].get(ag.index)
                        my_b_to_0 = b[ag.index].get(0)
                        if b0_to_me is not None and my_b_to_0 is not None:
                            h0_approx = ag._estimate_heading_from_map(b[0])
                            ag.update_heading_from_ref(h0_approx, b0_to_me, my_b_to_0)

                    # LVP
                    joy = JOYSTICK(ag.control_tick)
                    u_chassis = ag.hybrid_lvp(ag.dist_store, joy)
                    ag.sim.apply_control({ag.index: u_chassis}, SPEED_LIMIT)
                    ag.control_tick += 1

            # запись в таблицу
            ag.iteration += 1
            while ag.iteration >= len(ag.table):
                ag.table.append([None] * N)
            ag.table[ag.iteration][ag.index] = ag.theta.copy()

            # decay
            if cur_iter > 0 and cur_iter % DECAY_EVERY == 0:
                ag.alpha *= ALPHA_DECAY
                ag.beta *= BETA_DECAY

            ag.recv_store.pop(cur_iter, None)
            ag.recv_dist.pop(cur_iter, None)

        # ------------------------------------------------------------------
        # SPSA
        # ------------------------------------------------------------------
        def _observation(self, x_eval, target, present, neighbors_dist):
            ag: "SPSAFormationAgent" = self.agent
            self_pos = ag.theta[ag.index]
            total, count = 0.0, 0
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
            if d_it and d_it > 0.0:
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
                         - float(np.dot(self_pos, self_pos))
                         + d_it ** 2 - d_st ** 2)
                    r = float(np.dot(C, x_eval)) - D
                    total += r * r / Cn
                    count += 1
            return total / max(count, 1)

        def _spsa_step(self, present, neighbors_theta, neighbors_dist):
            ag: "SPSAFormationAgent" = self.agent
            new_theta = {j: ag.theta[j].copy() for j in range(N)}
            for target in [ag.index] + present:
                grad = np.zeros(DIMENSIONS)
                for _ in range(N_GRADIENT_SAMPLES):
                    signs = [1 if random.random() < 0.5 else -1
                             for _ in range(DIMENSIONS)]
                    delta = np.array([s * ag.Delta_abs for s in signs])
                    yp = self._observation(ag.theta[target] + ag.beta * delta,
                                           target, present, neighbors_dist)
                    ym = self._observation(ag.theta[target] - ag.beta * delta,
                                           target, present, neighbors_dist)
                    grad += delta * (yp - ym) / (2 * ag.beta)
                grad /= N_GRADIENT_SAMPLES
                consensus = np.zeros(DIMENSIONS)
                for nb in present:
                    nb_est = np.array(neighbors_theta[nb][str(target)], dtype=float)
                    consensus += nb_est - ag.theta[target]
                consensus *= ag.gamma
                new_theta[target] = (ag.theta[target]
                                     - ag.alpha * grad + ag.alpha * consensus)
            return new_theta

    class ReceiverBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=0.1)
            if msg:
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

    async def setup(self):
        tmpl = Template()
        tmpl.set_metadata("performative", "inform")
        self.add_behaviour(self.ReceiverBehaviour(), tmpl)
        self.add_behaviour(self.SendAndUpdateBehaviour(period=PERIOD))
