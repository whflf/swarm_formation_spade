# ── agents & network ────────────────────────────────────────────────────────
N = 4  # number of robots

AGENTS_CREDENTIALS = [
    ("agent0@localhost", "pass0"),
    ("agent1@localhost", "pass1"),
    ("agent2@localhost", "pass2"),
    ("agent3@localhost", "pass3"),
]

CENTER_JID = "center@localhost"

# Full K4 graph: every robot talks to all others
NEIGHBORS = [
    [1, 2, 3],
    [0, 2, 3],
    [0, 1, 3],
    [0, 1, 2],
]

# ── phase timing ─────────────────────────────────────────────────────────────
WARMUP_ITERS  = 8000  # pure-SPSA iterations before motion starts
CONTROL_ITERS = 300   # formation-control ticks after warmup
ITERATIONS    = WARMUP_ITERS + CONTROL_ITERS
SPSA_PER_TICK = 80    # SPSA steps per control tick

PERIOD           = 0.01   # PeriodicBehaviour period (s)
PER_ITER_TIMEOUT = 0.5    # max wait for neighbour messages per iteration (s)
DECAY_EVERY      = 500    # decay schedule period (iterations)

# ── link reliability ─────────────────────────────────────────────────────────
LINK_PROBABILITY     = 1.0
AGENT_UP_PROBABILITY = 1.0

# ── SPSA parameters ──────────────────────────────────────────────────────────
ALPHA              = 0.001
BETA               = 1.0
GAMMA              = 0.4
ALPHA_DECAY        = 0.975
BETA_DECAY         = 1.025
N_GRADIENT_SAMPLES = 5
DIMENSIONS         = 2

# ── sensors ──────────────────────────────────────────────────────────────────
NOISE_SCALE   = 0.5   # UWB range noise σ (m)
BEARING_NOISE = 0.05  # STM32 bearing noise σ (rad)
AREA_SIZE     = 10.0
SEED          = 7

# ── heading EMA filters ──────────────────────────────────────────────────────
EMA_H0  = 0.05   # robot-0 heading (map-based)
EMA_REL = 0.10   # robots 1..N-1 heading (bearing-relative)

# ── formation & control ──────────────────────────────────────────────────────
FORMATION_SIDE = 2.0   # side of the target square (m)
SPEED_LIMIT    = 0.02  # max chassis speed (m/tick)
GAMMA_LVP      = 0.08  # LVP controller gain


def joystick(tick: int) -> list[float]:
    """External velocity disturbance profile for robustness testing."""
    if tick < 50:
        return [0.00, 0.00]
    if tick < 120:
        return [0.02, 0.00]
    if tick < 200:
        return [0.01, 0.02]
    if tick < 250:
        return [-0.02, 0.00]
    return [0.00, 0.00]
