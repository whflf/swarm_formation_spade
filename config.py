"""
config.py — конфигурация SPSA-консенсуса + гибридного LVP на SPADE.
"""

# ------------------- агенты и сеть -------------------
N = 4   # квадрат

AGENTS_CREDENTIALS = [
    ("agent0@localhost", "pass0"),
    ("agent1@localhost", "pass1"),
    ("agent2@localhost", "pass2"),
    ("agent3@localhost", "pass3"),
]

CENTER_JID = "center@localhost"

# ------------------- синхронизация -------------------
WARMUP_ITERS     = 8000   # итераций чистого SPSA до начала движения
CONTROL_ITERS    = 300    # тиков управления после прогрева
ITERATIONS       = WARMUP_ITERS + CONTROL_ITERS
SPSA_PER_TICK    = 80     # шагов SPSA внутри одного тика управления

PERIOD           = 0.01
PER_ITER_TIMEOUT = 0.5
DECAY_EVERY      = 500

# ------------------- топология/помехи -------------------
LINK_PROBABILITY     = 1.0
AGENT_UP_PROBABILITY = 1.0

# ------------------- SPSA-параметры -------------------
ALPHA              = 0.001
BETA               = 1.0
GAMMA              = 0.4
ALPHA_DECAY        = 0.975
BETA_DECAY         = 1.025
N_GRADIENT_SAMPLES = 5
DIMENSIONS         = 2

# ------------------- датчики -------------------
NOISE_SCALE   = 0.5    # шум DW3000 (м)
BEARING_NOISE = 0.05   # шум STM32 (рад)
AREA_SIZE     = 10.0
SEED          = 7

# ------------------- heading -------------------
EMA_H0  = 0.05
EMA_REL = 0.10

# ------------------- формация (квадрат, сторона 2 м) -------------------
FORMATION_SIDE = 2.0
SPEED_LIMIT    = 0.02
GAMMA_LVP      = 0.08

def JOYSTICK(tick):
    if   0   <= tick < 50:   return [0.00,  0.00]
    elif 50  <= tick < 120:  return [0.02,  0.00]
    elif 120 <= tick < 200:  return [0.01,  0.02]
    elif 200 <= tick < 250:  return [-0.02, 0.00]
    else:                    return [0.00,  0.00]

# Полный граф K4
NEIGHBORS = [
    [1, 2, 3],
    [0, 2, 3],
    [0, 1, 3],
    [0, 1, 2],
]
