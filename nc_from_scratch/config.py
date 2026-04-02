SEED = 42

# Training hyperparameters
EPOCHS_XOR = 2000
LR_XOR = 0.1

EPOCHS_BINARY = 1000
LR_BINARY = 0.05

EPOCHS_REGRESSION = 1500
LR_REGRESSION = 0.01

BATCH_SIZE = 32

# Learning-rate scheduler settings
STEP_LR_BINARY_STEP_SIZE = 300
STEP_LR_BINARY_GAMMA = 0.5
STEP_LR_REG_STEP_SIZE = 400
STEP_LR_REG_GAMMA = 0.7

# Model save path prefix (without extension)
BINARY_MODEL_SAVE_PREFIX = "nn_from_scratch_binary_model"

# ReduceLROnPlateau settings (regression demo)
REDUCE_LR_FACTOR = 0.7
REDUCE_LR_PATIENCE = 30
REDUCE_LR_MIN_LR = 1e-5
REDUCE_LR_MIN_DELTA = 1e-4

# Early stopping settings (regression demo)
EARLY_STOPPING_PATIENCE = 80
EARLY_STOPPING_MIN_DELTA = 1e-4

