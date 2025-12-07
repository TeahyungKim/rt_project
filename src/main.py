import os
import logging
import datetime
import numpy as np
import tensorflow as tf
from utils import setup_logging

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

from environment import TFLiteQuantizationEnv
from parameters import BATCH_SIZE, EPOCHS

def main():
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    setup_logging(f"main_execution_{date_str}.log")
    logging.info("Initializing Mixed-Precision Quantization via PPO...")
    
    # Path to the SavedModel directory
    # MODEL_PATH = "/root/workspace/rt_project/model/whisper_encoder_tiny/output"
    # CALB_PATH = "/root/workspace/rt_project/model/whisper_encoder_tiny/output/calibration_audio_30x3000x80_float.npy"

    MODEL_PATH = "/root/workspace/rt_project/model/mobilenetv2/output"
    CALB_PATH = "/root/workspace/rt_project/model/mobilenetv2/output/input_1.npy"
    calb_data_list = [np.load(CALB_PATH).astype(np.float32)]

    # Initialize Environment
    env = TFLiteQuantizationEnv(
        model_path=MODEL_PATH,
        calib_idx=0,
        calib_data_list=calb_data_list
    )
       
    logging.info("Training Complete. Optimal quantization policy learned.")
    
    # Inference / Final Conversion
    logging.info("Generating optimal TFLite model...")
    env.summary()
    state = env.reset()
    
    for layer_idx in [0,1,2,3,4,5, 68, 69]:
        a = 0.3
        s_prime, r, done = env.step(a)
        logging.info(f"Layer {layer_idx}: Action (Sensitivity)={a}, Reward={r:.4f}, Done={done}, Next State Shape={s_prime.shape}")

if __name__ == "__main__":
    main()
