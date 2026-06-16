import numpy as np
from typing import Dict, Any
from enum import Enum

from numpy import dtype, float64, ndarray


class NoiseType(Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    PARETO = "pareto"


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class RobotSwarmSimulator:
    """
    Симулятор роя роботов.
    Знает истинные позиции, генерирует зашумлённые расстояния для SPSA,
    пеленги (STM32) и применяет управление (меканум-колёса).
    """

    def __init__(
            self,
            num_robots: int,
            noise_type: NoiseType = NoiseType.UNIFORM,
            noise_scale: float = 0.5,
            area_size: float = 10.0,
            max_comm_range: float = None,
            bearing_noise: float = 0.05,
            seed: int = None,
    ):
        self.num_robots = num_robots
        self.robot_ids = list(range(num_robots))
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.max_comm_range = max_comm_range
        self.bearing_noise = bearing_noise

        if seed is not None:
            np.random.seed(seed)

        self.true_positions: Dict[int, np.ndarray] = {
            i: np.random.uniform(0, area_size, size=2)
            for i in self.robot_ids
        }

        # истинные курсы корпусов при старте
        rng = np.random.default_rng(seed + 1 if seed is not None else None)
        self.true_heading: Dict[int, float] = {
            i: rng.uniform(-np.pi, np.pi) for i in self.robot_ids
        }

    # ------------------------------------------------------------------
    # Расстояния
    # ------------------------------------------------------------------

    def _true_distance(self, i: int, j: int) -> float:
        diff = self.true_positions[i] - self.true_positions[j]
        return float(np.linalg.norm(diff))

    def _generate_noise(self) -> float | ndarray[tuple[Any, ...], dtype[float64]] | None:
        if self.noise_type == NoiseType.UNIFORM:
            return np.random.uniform(-self.noise_scale, self.noise_scale)
        elif self.noise_type == NoiseType.GAUSSIAN:
            return np.random.normal(0, self.noise_scale)
        elif self.noise_type == NoiseType.PARETO:
            alpha, m = 10, self.noise_scale
            sample = (np.random.pareto(alpha) + 1) * m
            mean = m * alpha / (alpha - 1)
            return sample - mean
        return None

    def _can_communicate(self, i: int, j: int) -> bool:
        if self.max_comm_range is None:
            return True
        return self._true_distance(i, j) <= self.max_comm_range

    def get_noisy_distances(self) -> Dict[int, Dict[int, float]]:
        """Зашумлённые расстояния (DW3000)."""
        distances = {}
        for i in self.robot_ids:
            distances[i] = {}
            for j in self.robot_ids:
                if i == j or not self._can_communicate(i, j):
                    continue
                true_d = self._true_distance(i, j)
                distances[i][j] = max(0.0, true_d + self._generate_noise())
        return distances

    # ------------------------------------------------------------------
    # Пеленги (STM32)
    # ------------------------------------------------------------------

    def measure_bearings(self) -> Dict[int, Dict[int, float]]:
        """Пеленги на соседей в СК корпуса каждого робота."""
        b = {}
        for i in self.robot_ids:
            b[i] = {}
            for j in self.robot_ids:
                if i == j:
                    continue
                diff = self.true_positions[j] - self.true_positions[i]
                abs_angle = np.arctan2(diff[1], diff[0])
                phi = _wrap(abs_angle - self.true_heading[i])
                b[i][j] = _wrap(phi + np.random.normal(0, self.bearing_noise))
        return b

    # ------------------------------------------------------------------
    # Управление (меканум-колёса)
    # ------------------------------------------------------------------

    def apply_control(self, chassis_controls: Dict[int, np.ndarray],
                      speed_limit: float):
        """
        Применяет управление в СК корпуса каждого робота.
        """
        for i in self.robot_ids:
            if i not in chassis_controls:
                continue
            v = np.array(chassis_controls[i], dtype=float)
            s = np.linalg.norm(v)
            if s > speed_limit:
                v = v * (speed_limit / s)
            h = self.true_heading[i]
            R = np.array([[np.cos(h), -np.sin(h)],
                          [np.sin(h), np.cos(h)]])
            self.true_positions[i] = self.true_positions[i] + R @ v

    # ------------------------------------------------------------------
    # Движение (тест динамического сценария)
    # ------------------------------------------------------------------

    def move_robot(self, robot_id: int, velocity: np.ndarray, dt: float = 0.1):
        self.true_positions[robot_id] += velocity * dt

    def move_all(self, velocities: Dict[int, np.ndarray], dt: float = 0.1):
        for i, v in velocities.items():
            self.move_robot(i, v, dt)

    # ------------------------------------------------------------------
    # Метрики
    # ------------------------------------------------------------------

    def compute_estimation_error(self, theta_hat):
        errors = {}
        for i in self.robot_ids:
            errors[i] = {}
            true_self = self.true_positions[i]
            for j in self.robot_ids:
                if i == j or j not in theta_hat.get(i, {}):
                    continue
                true_relative = self.true_positions[j] - true_self
                estimated_relative = theta_hat[i][j] - theta_hat[i][i]
                errors[i][j] = float(np.linalg.norm(true_relative - estimated_relative))
        return errors

    def mean_error(self, theta_hat, adjacency=None) -> float:
        errors = self.compute_estimation_error(theta_hat)
        all_errors = []
        for i, row in errors.items():
            for j, e in row.items():
                if adjacency is None or adjacency[i][j] == 1:
                    all_errors.append(e)
        return float(np.mean(all_errors)) if all_errors else 0.0

    def consensus_error(self, theta_hat) -> float:
        total, count = 0.0, 0
        for j in self.robot_ids:
            estimates = [theta_hat[i][j] for i in self.robot_ids
                         if i != j and j in theta_hat.get(i, {})]
            if len(estimates) < 2:
                continue
            mean_est = np.mean(estimates, axis=0)
            for est in estimates:
                total += float(np.linalg.norm(est - mean_est))
                count += 1
        return total / count if count else 0.0

    def heading_spread(self, heading_estimates: dict) -> float:
        errs = [_wrap(heading_estimates[i] - self.true_heading[i])
                for i in self.robot_ids]
        return float(np.std(errs))
