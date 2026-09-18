from enum import Enum

import numpy as np


class NoiseType(Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    PARETO = "pareto"


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class RobotSwarmSimulator:
    """Physics simulator: ground-truth positions, noisy sensors, mecanum control."""

    def __init__(
        self,
        num_robots: int,
        noise_type: NoiseType = NoiseType.UNIFORM,
        noise_scale: float = 0.5,
        area_size: float = 10.0,
        max_comm_range: float | None = None,
        bearing_noise: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.num_robots = num_robots
        self.robot_ids = list(range(num_robots))
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.max_comm_range = max_comm_range
        self.bearing_noise = bearing_noise

        if seed is not None:
            np.random.seed(seed)

        self.true_positions: dict[int, np.ndarray] = {
            i: np.random.uniform(0, area_size, size=2) for i in self.robot_ids
        }

        rng = np.random.default_rng(seed + 1 if seed is not None else None)
        self.true_heading: dict[int, float] = {
            i: rng.uniform(-np.pi, np.pi) for i in self.robot_ids
        }

    # ── distances ────────────────────────────────────────────────────────────

    def _true_distance(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.true_positions[i] - self.true_positions[j]))

    def _generate_noise(self) -> float:
        if self.noise_type == NoiseType.UNIFORM:
            return float(np.random.uniform(-self.noise_scale, self.noise_scale))
        if self.noise_type == NoiseType.GAUSSIAN:
            return float(np.random.normal(0, self.noise_scale))
        # Pareto (heavy-tailed), zero-mean
        alpha, m = 10, self.noise_scale
        sample = (np.random.pareto(alpha) + 1) * m
        return float(sample - m * alpha / (alpha - 1))

    def _can_communicate(self, i: int, j: int) -> bool:
        if self.max_comm_range is None:
            return True
        return self._true_distance(i, j) <= self.max_comm_range

    def get_noisy_distances(self) -> dict[int, dict[int, float]]:
        """Simulated UWB (DW3000) range measurements."""
        distances: dict[int, dict[int, float]] = {}
        for i in self.robot_ids:
            distances[i] = {}
            for j in self.robot_ids:
                if i == j or not self._can_communicate(i, j):
                    continue
                distances[i][j] = max(0.0, self._true_distance(i, j) + self._generate_noise())
        return distances

    # ── bearings ─────────────────────────────────────────────────────────────

    def measure_bearings(self) -> dict[int, dict[int, float]]:
        """Simulated STM32 phasor-array bearing measurements (body frame)."""
        b: dict[int, dict[int, float]] = {}
        for i in self.robot_ids:
            b[i] = {}
            for j in self.robot_ids:
                if i == j:
                    continue
                diff = self.true_positions[j] - self.true_positions[i]
                phi = _wrap(np.arctan2(diff[1], diff[0]) - self.true_heading[i])
                b[i][j] = _wrap(phi + np.random.normal(0, self.bearing_noise))
        return b

    # ── control ───────────────────────────────────────────────────────────────

    def apply_control(
        self,
        chassis_controls: dict[int, np.ndarray],
        speed_limit: float,
    ) -> None:
        """Apply body-frame velocity commands to all robots (speed-limited)."""
        for i in self.robot_ids:
            if i not in chassis_controls:
                continue
            v = np.array(chassis_controls[i], dtype=float)
            s = np.linalg.norm(v)
            if s > speed_limit:
                v *= speed_limit / s
            h = self.true_heading[i]
            R = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
            self.true_positions[i] = self.true_positions[i] + R @ v

    # ── metrics ───────────────────────────────────────────────────────────────

    def compute_estimation_error(self, theta_hat: dict) -> dict[int, dict[int, float]]:
        errors: dict[int, dict[int, float]] = {}
        for i in self.robot_ids:
            errors[i] = {}
            for j in self.robot_ids:
                if i == j or j not in theta_hat.get(i, {}):
                    continue
                true_rel = self.true_positions[j] - self.true_positions[i]
                est_rel = theta_hat[i][j] - theta_hat[i][i]
                errors[i][j] = float(np.linalg.norm(true_rel - est_rel))
        return errors

    def mean_error(self, theta_hat: dict, adjacency: dict | None = None) -> float:
        errors = self.compute_estimation_error(theta_hat)
        all_errors = [
            e
            for i, row in errors.items()
            for j, e in row.items()
            if adjacency is None or adjacency[i][j] == 1
        ]
        return float(np.mean(all_errors)) if all_errors else 0.0

    def consensus_error(self, theta_hat: dict) -> float:
        total, count = 0.0, 0
        for j in self.robot_ids:
            estimates = [
                theta_hat[i][j]
                for i in self.robot_ids
                if i != j and j in theta_hat.get(i, {})
            ]
            if len(estimates) < 2:
                continue
            mean_est = np.mean(estimates, axis=0)
            for est in estimates:
                total += float(np.linalg.norm(est - mean_est))
                count += 1
        return total / count if count else 0.0

    def heading_spread(self, heading_estimates: dict) -> float:
        errs = [_wrap(heading_estimates[i] - self.true_heading[i]) for i in self.robot_ids]
        return float(np.std(errs))
