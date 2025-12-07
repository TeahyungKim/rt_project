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
from ppo import PPO
from parameters import EPOCHS, CALIB_SIZE

def main():
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    setup_logging(f"train_execution_{date_str}.log")
    logging.info("Initializing Mixed-Precision Quantization via PPO...")
    
    # Path to the SavedModel directory (Using Encoder Tiny as per main.py example)
    # MODEL_PATH = "/root/workspace/rt_project/model/whisper_encoder_tiny/output"
    # CALB_PATH = "/root/workspace/rt_project/model/whisper_encoder_tiny/output/calibration_audio_30x3000x80_float.npy"
    MODEL_PATH = "/root/workspace/rt_project/model/mobilenetv2/output"
    CALB_PATH = "/root/workspace/rt_project/model/mobilenetv2/output/input_1.npy"

    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model path {MODEL_PATH} not found.")
        return

    # Load Calibration Data
    if os.path.exists(CALB_PATH):
        calb_data_list = [np.load(CALB_PATH).astype(np.float32)]
    else:
        logging.error(f"Calibration data {CALB_PATH} not found. Generating random data.")
        # Fallback to random data matching input shape [1, 80, 3000] or similar?
        # Encoder tiny input is [1, 80, 3000] usually, but let's assume the user has the file or we fail.
        # Based on main.py, it seems the file exists.
        return

    
    # Initialize Agent
    agent = PPO() 
    
    # Tracking Metrics
    episode_rewards = []
    
    logging.info(f"Starting training for {EPOCHS} episodes...")

    # prepare multiple environments depending on calibration data
    envs = []
    for i in range(CALIB_SIZE):
        envs.append(TFLiteQuantizationEnv(
            model_path=MODEL_PATH,
            calib_idx=i,
            calib_data_list=calb_data_list
        ))

    for ep in range(EPOCHS):        
        # Initialize Environment
        env = envs[ep % CALIB_SIZE]
        state = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            # 1. Get Action
            # Add batch dimension [1, state_dim]
            tf_state = tf.convert_to_tensor([state], dtype=tf.float32)
            mu, sigma = agent.pi(tf_state)
            
            # Numpy Sampling
            mu_np = mu.numpy()[0]
            sigma_np = sigma.numpy()[0] + 1e-5
            
            a = np.random.normal(mu_np, sigma_np)
            log_prob_a = -0.5 * ((a - mu_np) / sigma_np)**2 - np.log(sigma_np) - 0.5 * np.log(2 * np.pi)
            
            # Clip action for environment
            a_env = np.clip(a, 0.0, 1.0).item()
            
            # Convert to scalars for storage
            a_scalar = a.item()
            log_prob_a_scalar = log_prob_a.item()
            
            logging.debug(f"Episode {ep+1} | State: {state} | Mu: {mu_np} | Sigma: {sigma_np} | Action: {a_scalar}")

            # 2. Step Environment
            next_state, reward, done = env.step(a_env)
            
            # 3. Store Experience
            # PPO expects: (s, a, r, s_prime, log_prob_a, done)
            # Store unclipped 'a' to maintain Gaussian consistency in PPO update
            agent.put_data((state, a_scalar, reward, next_state, log_prob_a_scalar, done))
            
            state = next_state
            ep_reward += reward
            
        # 4. Train Agent at end of episode
        agent.train_net()
        
        episode_rewards.append(ep_reward)
        
        if (ep + 1) % 1 == 0:
            logging.info(f"Episode {ep+1}/{EPOCHS} | Total Reward: {ep_reward:.4f}")

    logging.info("Training Complete.")
    
    # Save the trained policy if needed
    # agent.save_weights("ppo_agent_weights")

if __name__ == "__main__":
    main()
