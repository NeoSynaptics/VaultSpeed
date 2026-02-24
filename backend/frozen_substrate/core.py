import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def roll_no_wrap(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift using np.roll but zero out wrap-around artifacts."""
    b = np.roll(a, shift=(dy, dx), axis=(0, 1))
    if dy > 0:
        b[:dy, :] = 0.0
    elif dy < 0:
        b[dy:, :] = 0.0
    if dx > 0:
        b[:, :dx] = 0.0
    elif dx < 0:
        b[:, dx:] = 0.0
    return b


def laplacian4(a: np.ndarray) -> np.ndarray:
    """4-neighbor Laplacian with zero boundary."""
    up = roll_no_wrap(a, -1, 0)
    dn = roll_no_wrap(a, 1, 0)
    lf = roll_no_wrap(a, 0, -1)
    rt = roll_no_wrap(a, 0, 1)
    return (up + dn + lf + rt) - 4.0 * a


def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------------
# Frozen Core v3
# -----------------------------

class FrozenCoreV3:
    """Frozen Core v3 with an emergent coordination field.

    Notes:
    - Uses 8-neighbor local synapses (radius 1) for tractability.
    - Coordination Field C is driven by *surprise*: positive deviation of |error| above a running baseline.
    - One source (C) -> three projections (fast/mid/slow) via different EMA time constants.
    - Field is modulatory: it gates coupling/plasticity/pruning; it does not act as a separate hard "input current".
    """

    # 8-neighbor directions (dy, dx) and distance
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(
        self,
        H=40,
        W=40,
        seed=42,
        # External drive channel
        enable_external_drive=True,
        # Dynamics (FAST)
        x_bound=1.0,
        leak=0.18,
        neighbor_gain=0.55,
        noise_scale=0.01,
        # Predictor
        a_init=0.9,
        a_bounds=(-1.2, 1.2),
        # Plasticity (SLOW)
        w_init_scale=0.10,
        w_budget=3.0,
        w_decay=0.0015,
        w_lr=0.004,
        a_lr=0.0006,
        plastic_every=10,
        # Survival (SLOWEST)
        s_init=1.0,
        s_max=2.0,
        alive_min=0.02,
        survive_gain=0.006,
        survive_drain=0.006,
        metabolic_cost=0.001,
        prune_every=250,
        prune_grace=800,
        # Pacemakers
        pacemakers=((10, 10), (30, 28)),
        pace_amplitude=0.50,
        pace_frequency=0.10,
        pace_jitter=0.03,
        enable_pacemakers=False,
        # Coordination Field (INTERMEDIATE)
        field_decay=0.06,
        field_diffusion=0.20,
        vote_gain=0.35,
        surprise_baseline_lr=0.003,
        field_threshold=0.10,
        field_k=10.0,
        # Projections (EMA on A)
        A_fast_tau=0.15,
        A_mid_tau=0.03,
        A_slow_tau=0.01,
        # Modulation strengths
        mod_coupling=0.25,
        mod_plasticity=1.50,
        mod_pruning=1.00,
    ):
        self.rng = np.random.default_rng(seed)
        self.H, self.W = H, W

        # Exogenous drive buffer (cleared each step).
        self.enable_external_drive = bool(enable_external_drive)
        self.drive = np.zeros((H, W), dtype=np.float32)

        # State
        self.x = np.zeros((H, W), dtype=np.float32)
        self.x_prev = np.zeros((H, W), dtype=np.float32)

        self.a = np.full((H, W), a_init, dtype=np.float32)
        self.a_bounds = a_bounds

        self.s = np.full((H, W), s_init, dtype=np.float32)
        self.s_max = float(s_max)
        self.alive = np.ones((H, W), dtype=np.float32)

        # Local weights per direction
        w = self.rng.normal(0.0, w_init_scale, size=(H, W, len(self.DIRS))).astype(np.float32)
        self.w_budget = float(w_budget)
        self.w = self._enforce_budget(w)

        # Error bookkeeping
        self.e = np.zeros((H, W), dtype=np.float32)
        self.e_prev = np.zeros((H, W), dtype=np.float32)

        # Metabolic / fatigue-like trace (slow EMA of |x|)
        self.fatigue = np.zeros((H, W), dtype=np.float32)
        self.fatigue_tau = 0.02

        # Dynamics params
        self.x_bound = float(x_bound)
        self.leak = float(leak)
        self.neighbor_gain = float(neighbor_gain)
        self.noise_scale = float(noise_scale)

        # Plasticity params
        self.w_decay = float(w_decay)
        self.w_lr = float(w_lr)
        self.a_lr = float(a_lr)
        self.plastic_every = int(plastic_every)

        # Survival params
        self.alive_min = float(alive_min)
        self.survive_gain = float(survive_gain)
        self.survive_drain = float(survive_drain)
        self.metabolic_cost = float(metabolic_cost)
        self.prune_every = int(prune_every)
        self.prune_grace = int(prune_grace)

        # Pacemakers
        self.pacemakers = [(y, x) for (y, x) in pacemakers]
        self.pace_amp = float(pace_amplitude)
        self.pace_freq = float(pace_frequency)
        self.pace_jitter = float(pace_jitter)
        self.enable_pacemakers = bool(enable_pacemakers)

        # Coordination field
        self.C = np.zeros((H, W), dtype=np.float32)
        self.field_decay = float(field_decay)
        self.field_diffusion = float(field_diffusion)
        self.vote_gain = float(vote_gain)

        # Surprise baseline per neuron (EMA of |e|)
        self.err_baseline = np.zeros((H, W), dtype=np.float32)
        self.surprise_baseline_lr = float(surprise_baseline_lr)

        # Activation from field
        self.field_threshold = float(field_threshold)
        self.field_k = float(field_k)

        # Projections (EMA) on A
        self.A_fast = np.zeros((H, W), dtype=np.float32)
        self.A_mid = np.zeros((H, W), dtype=np.float32)
        self.A_slow = np.zeros((H, W), dtype=np.float32)
        self.A_fast_tau = float(A_fast_tau)
        self.A_mid_tau = float(A_mid_tau)
        self.A_slow_tau = float(A_slow_tau)

        # Modulation
        self.mod_coupling = float(mod_coupling)
        self.mod_plasticity = float(mod_plasticity)
        self.mod_pruning = float(mod_pruning)

        # Time
        self.t = 0

        # Traces
        self.trace_mean_abs_e = []
        self.trace_alive_frac = []
        self.trace_mean_C = []
        self.trace_mean_A = []
        self.trace_mean_s = []
        self.trace_mean_fatigue = []

    def reset(self, seed=None):
        """Reset state while keeping the same hyperparameters."""
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        self.x.fill(0.0)
        self.x_prev.fill(0.0)
        self.drive.fill(0.0)

        self.e.fill(0.0)
        self.e_prev.fill(0.0)
        self.fatigue.fill(0.0)

        self.C.fill(0.0)
        self.err_baseline.fill(0.0)
        self.A_fast.fill(0.0)
        self.A_mid.fill(0.0)
        self.A_slow.fill(0.0)

        self.s.fill(self.s_max * 0.5)
        self.alive.fill(1.0)

        self.t = 0

        self.trace_mean_abs_e.clear()
        self.trace_alive_frac.clear()
        self.trace_mean_C.clear()
        self.trace_mean_A.clear()
        self.trace_mean_s.clear()
        self.trace_mean_fatigue.clear()

    def add_drive(self, field: np.ndarray, gain: float = 1.0):
        """Accumulate an external drive field (same shape as x).

        The driver script can call this multiple times per step.
        The core consumes and clears the accumulated drive during step().
        """
        if not self.enable_external_drive:
            return
        if field.shape != (self.H, self.W):
            raise ValueError(f"drive field must have shape {(self.H, self.W)}, got {field.shape}")
        if gain != 1.0:
            self.drive += (gain * field).astype(np.float32)
        else:
            self.drive += field.astype(np.float32)

    def _enforce_budget(self, w: np.ndarray) -> np.ndarray:
        abs_sum = np.sum(np.abs(w), axis=-1, keepdims=True)
        scale = np.ones_like(abs_sum)
        mask = abs_sum > 1e-8
        scale[mask] = np.minimum(1.0, self.w_budget / abs_sum[mask])
        return w * scale

    def _pacemaker_drive(self) -> np.ndarray:
        if not self.enable_pacemakers:
            return np.zeros((self.H, self.W), dtype=np.float32)
        drive = np.zeros((self.H, self.W), dtype=np.float32)
        base = self.pace_amp * np.sin(2.0 * np.pi * self.pace_freq * self.t)
        for (y, x) in self.pacemakers:
            jitter = self.rng.normal(0.0, self.pace_jitter)
            drive[y, x] = base + jitter
        return drive

    def step(self):
        # Save previous
        self.x_prev[...] = self.x
        self.e_prev[...] = self.e

        # 1) Compute neighbor drive (8-neighbor weighted sum)
        pre = (self.x_prev * self.alive).astype(np.float32)
        neigh_sum = np.zeros((self.H, self.W), dtype=np.float32)
        for k, (dy, dx) in enumerate(self.DIRS):
            neigh_sum += self.w[..., k] * roll_no_wrap(pre, dy, dx)

        # 2) Coordination modulation (fast) affects coupling slightly
        g_couple = 1.0 + self.mod_coupling * self.A_fast

        # 3) Metabolic / fatigue update (slow)
        absx = np.abs(self.x_prev)
        self.fatigue = (1.0 - self.fatigue_tau) * self.fatigue + self.fatigue_tau * absx

        # 4) Update state dynamics
        noise = self.rng.normal(0.0, self.noise_scale, size=(self.H, self.W)).astype(np.float32)
        pacer = self._pacemaker_drive()

        # Consume external drive and clear it
        if self.enable_external_drive:
            drive = self.drive.copy()
            self.drive.fill(0.0)
        else:
            drive = 0.0

        fatigue_damp = 1.0 - 0.35 * self.fatigue
        fatigue_damp = clamp(fatigue_damp, 0.50, 1.00)

        x_lin = (1.0 - self.leak) * self.x_prev + (self.neighbor_gain * g_couple) * neigh_sum + pacer + drive + noise
        x_lin *= fatigue_damp

        self.x = np.tanh(x_lin).astype(np.float32) * self.alive
        self.x = clamp(self.x, -self.x_bound, self.x_bound)

        # Hard safety: if anything went non-finite, zero it out
        if not np.isfinite(self.x).all():
            self.x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # 5) Local self-prediction error (after update)
        x_hat = (self.a * self.x_prev).astype(np.float32)
        self.e = (self.x - x_hat).astype(np.float32) * self.alive

        # 6) Surprise + Coordination Field update
        abs_e = np.abs(self.e)
        self.err_baseline = (1.0 - self.surprise_baseline_lr) * self.err_baseline + self.surprise_baseline_lr * abs_e
        surprise = np.maximum(0.0, abs_e - self.err_baseline) * self.alive

        # Field C: decay + diffusion + merit-based votes
        self.C = (1.0 - self.field_decay) * self.C + self.vote_gain * surprise
        self.C += self.field_diffusion * laplacian4(self.C)
        self.C = np.maximum(0.0, self.C).astype(np.float32)

        # Field activation A (nonlinear, thresholded)
        A = sigmoid(self.field_k * (self.C - self.field_threshold)).astype(np.float32)

        # Projections: fast/mid/slow EMAs
        self.A_fast = (1.0 - self.A_fast_tau) * self.A_fast + self.A_fast_tau * A
        self.A_mid = (1.0 - self.A_mid_tau) * self.A_mid + self.A_mid_tau * A
        self.A_slow = (1.0 - self.A_slow_tau) * self.A_slow + self.A_slow_tau * A

        # 7) Plasticity update (gated by A_mid)
        if self.t % self.plastic_every == 0:
            self.w *= (1.0 - self.w_decay)

            e2_prev = self.e_prev * self.e_prev
            e2 = self.e * self.e
            improvement = np.maximum(0.0, e2_prev - e2)

            gate = (1.0 + self.mod_plasticity * self.A_mid).astype(np.float32)

            self.a += (self.a_lr * gate) * (self.e * self.x_prev)
            self.a = clamp(self.a, self.a_bounds[0], self.a_bounds[1]).astype(np.float32)

            for k, (dy, dx) in enumerate(self.DIRS):
                presyn = roll_no_wrap(pre, dy, dx)
                delta = (self.w_lr * gate) * improvement * presyn
                self.w[..., k] += delta

            self.w = self._enforce_budget(self.w)

        # 8) Survival update (slow)
        useful = (1.0 - clamp(abs_e, 0.0, 1.0)) * (0.5 + 0.5 * np.abs(self.x))
        drain = abs_e + 0.5 * (np.abs(self.x) < 0.02).astype(np.float32)

        prune_gate = (1.0 + self.mod_pruning * self.A_slow).astype(np.float32)

        self.s += self.survive_gain * useful
        self.s -= self.survive_drain * drain * prune_gate
        self.s -= self.metabolic_cost * self.fatigue
        self.s = clamp(self.s, 0.0, self.s_max).astype(np.float32)

        # 9) Pruning
        if (self.t > self.prune_grace) and (self.t % self.prune_every == 0):
            alive_mask = self.alive > 0.0
            alive_count = int(np.sum(alive_mask))
            if alive_count > 0:
                target_min = int(self.alive_min * self.H * self.W)
                flat_s = self.s[alive_mask]
                q = 0.02
                thr = float(np.quantile(flat_s, q)) if flat_s.size > 1 else float(flat_s[0])
                kill = (self.s <= thr) & alive_mask

                if alive_count - int(np.sum(kill)) < target_min:
                    needed = alive_count - target_min
                    if needed <= 0:
                        kill[...] = False
                    else:
                        idx = np.argsort(flat_s)
                        if needed < idx.size:
                            keep_kill_thr = float(flat_s[idx[needed - 1]])
                            kill = (self.s <= keep_kill_thr) & alive_mask

                self.alive[kill] = 0.0
                self.x[kill] = 0.0
                self.x_prev[kill] = 0.0
                self.e[kill] = 0.0
                self.C[kill] = 0.0
                self.A_fast[kill] = 0.0
                self.A_mid[kill] = 0.0
                self.A_slow[kill] = 0.0

        # 10) Traces
        alive_frac = float(np.mean(self.alive))
        self.trace_alive_frac.append(alive_frac)
        self.trace_mean_abs_e.append(float(np.mean(abs_e[self.alive > 0.0])) if alive_frac > 0 else 0.0)
        self.trace_mean_C.append(float(np.mean(self.C)))
        self.trace_mean_A.append(float(np.mean(A)))
        self.trace_mean_s.append(float(np.mean(self.s[self.alive > 0.0])) if alive_frac > 0 else 0.0)
        self.trace_mean_fatigue.append(float(np.mean(self.fatigue[self.alive > 0.0])) if alive_frac > 0 else 0.0)

        self.t += 1

    def run(self, steps=5000, change_pace_at=None, new_pace_freq=None):
        for _ in range(int(steps)):
            if change_pace_at is not None and new_pace_freq is not None:
                if self.t == int(change_pace_at):
                    self.pace_freq = float(new_pace_freq)
                    print(f"[t={self.t}] pacemaker frequency changed to {self.pace_freq}")
            self.step()

    def plot(self, every=1):
        t = np.arange(0, len(self.trace_mean_abs_e))
        fig = plt.figure(figsize=(10, 7))
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(t[::every], np.array(self.trace_mean_abs_e)[::every])
        ax1.set_ylabel("mean |error| (alive)")
        ax1.grid(True, alpha=0.2)

        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(t[::every], np.array(self.trace_alive_frac)[::every])
        ax2.set_ylabel("alive fraction")
        ax2.grid(True, alpha=0.2)

        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(t[::every], np.array(self.trace_mean_C)[::every], label="mean C")
        ax3.plot(t[::every], np.array(self.trace_mean_A)[::every], label="mean A")
        ax3.plot(t[::every], np.array(self.trace_mean_s)[::every], label="mean survival")
        ax3.set_ylabel("field / survival")
        ax3.set_xlabel("t")
        ax3.grid(True, alpha=0.2)
        ax3.legend(loc="best")

        plt.tight_layout()
        plt.show()

    def snapshot(self):
        return {
            "x": self.x.copy(),
            "alive": self.alive.copy(),
            "error": self.e.copy(),
            "C": self.C.copy(),
            "A_fast": self.A_fast.copy(),
            "A_mid": self.A_mid.copy(),
            "A_slow": self.A_slow.copy(),
            "survival": self.s.copy(),
            "fatigue": self.fatigue.copy(),
        }


def demo():
    core = FrozenCoreV3(H=40, W=40, seed=7)
    core.run(steps=4000, change_pace_at=2500, new_pace_freq=0.14)
    snap = core.snapshot()

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(snap["x"], interpolation="nearest")
    plt.title("x")
    plt.colorbar(fraction=0.046)

    plt.subplot(2, 3, 2)
    plt.imshow(snap["error"], interpolation="nearest")
    plt.title("error")
    plt.colorbar(fraction=0.046)

    plt.subplot(2, 3, 3)
    plt.imshow(snap["C"], interpolation="nearest")
    plt.title("field C")
    plt.colorbar(fraction=0.046)

    plt.subplot(2, 3, 4)
    plt.imshow(snap["A_mid"], interpolation="nearest")
    plt.title("A_mid")
    plt.colorbar(fraction=0.046)

    plt.subplot(2, 3, 5)
    plt.imshow(snap["survival"], interpolation="nearest")
    plt.title("survival")
    plt.colorbar(fraction=0.046)

    plt.subplot(2, 3, 6)
    plt.imshow(snap["alive"], interpolation="nearest", vmin=0, vmax=1)
    plt.title("alive")
    plt.colorbar(fraction=0.046)

    plt.tight_layout()
    plt.show()

    core.plot(every=5)


if __name__ == "__main__":
    demo()
