import numpy as np

class VelocitySineGait:
    def __init__(self, dt=0.016):
        self.t = 0.0
        self.dt = dt

        # --- Gait timing ---
        self.period = 0.8                # seconds (slow walk)
        self.omega = 2.0 * np.pi / self.period

        # --- Velocity amplitudes (rad/s) ---
        self.A_hip = 0.20
        self.A_thigh = 0.25
        self.A_calf = 0.20

        # --- Phase offsets (crawl gait) ---
        self.phase_offsets = {
            "FR": 0.0,
            "BL": np.pi,
            "FL": np.pi / 2.0,
            "BR": 3.0 * np.pi / 2.0
        }

        self.leg_order = ["FR", "FL", "BR", "BL"]

    def reset(self):
        self.t = 0.0

    def step(self):
        self.t += self.dt
        phi = self.omega * self.t

        action = np.zeros(12, dtype=np.float32)

        for leg_idx, leg_name in enumerate(self.leg_order):
            φ_leg = phi + self.phase_offsets[leg_name]

            # --- Velocity profiles ---
            hip_vel = self.A_hip * self.omega * np.cos(φ_leg)

            thigh_raw = self.A_thigh * self.omega * np.cos(φ_leg)
            thigh_vel = max(0.0, thigh_raw)  # lift only

            calf_vel = 0.5 * thigh_vel       # softer foot clearance

            base = leg_idx * 3
            action[base + 0] = hip_vel
            action[base + 1] = thigh_vel
            action[base + 2] = calf_vel

        # IMPORTANT: clip to PPO/Godot expectations
        return np.clip(action, -2.0, 2.0)