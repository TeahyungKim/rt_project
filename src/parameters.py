# Hyperparameters
CALIB_SIZE = 30
BATCH_SIZE = 1
EPOCHS = 200
GAMMA = 0.99
LEARNING_RATE = 3e-4
CLIP_RATIO = 0.2
ENTROPY_COEF = 0.01
QUANTIZATION_THRESHOLD = 0.5  # Action threshold
LMBDA = 0.95
K_EPOCH = 3

# Reward Weights
W_COMPRESSION = 0.5
W_ACCURACY = 1.0
TARGET_MSE_THRESHOLD = 0.001

# Model and Calibration Paths
MODEL_PATH = "/root/workspace/rt_project/model/mobilenetv2/output"
CALB_PATH = "/root/workspace/rt_project/model/mobilenetv2/output/input_1.npy"

# Debug
ENABLE_DEBUG = True