import os
# Suppress TensorFlow logs (Must be before importing tensorflow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

from environment import TFLiteQuantizationEnv
# from ppo import PPO
from parameters import BATCH_SIZE, EPOCHS

def main():
    # Suppress TensorFlow logs
    tf.get_logger().setLevel(logging.ERROR)
    print("Initializing Mixed-Precision Quantization via PPO...")
    
    # Path to the SavedModel directory
    MODEL_PATH = "/root/workspace/rt_project/model/whisper_encoder_tiny/output"
    CALB_PATH = "/root/workspace/rt_project/model/whisper_encoder_tiny/output/calibration_audio_30x3000x80_float.npy"

    CALB_PATH = "/root/workspace/rt_project/model/whisper_encoder_tiny/output/calibration_audio_30x3000x80_float.npy"
    calb_data_list = [np.load(CALB_PATH).astype(np.float32)]

    # Initialize Environment
    env = TFLiteQuantizationEnv(
        model_path=MODEL_PATH,
        calib_idx=0,
        calib_data_list=calb_data_list
    )
       
    print("Training Complete. Optimal quantization policy learned.")
    
    # Inference / Final Conversion
    print("Generating optimal TFLite model...")
    env.summary()
    state = env.reset()
    # a = model.fi(state)
    for layer_idx in [0,1,2,3,4,5]:
        # if layer_idx in (0, 1,2,3,4,5):
        #     state = env._test(layer_idx, True)
        # else:
        #     state = env._test(layer_idx, False)
        a = 0.3
        if layer_idx in (0,1,2,3,4):
            a = 0.7
        s_prime, r, done = env.step(a)
        print(f"Layer {layer_idx}: Action (Sensitivity)={a}, Reward={r:.4f}, Done={done}, Next State Shape={s_prime.shape}")
if __name__ == "__main__":
    main()
