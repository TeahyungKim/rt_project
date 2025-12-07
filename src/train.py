import os
import logging
import datetime
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import setup_logging


# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

from environment import TFLiteQuantizationEnv
from ppo import PPO
from parameters import EPOCHS, CALIB_SIZE, MODEL_PATH, CALB_PATH

def plot_training_results(ep, EPOCHS, all_epoch_rewards, all_epoch_quant_counts, all_epoch_avg_mses, step_mses, step_quant_counts, step_rmse_scales):
    """
    Plots training results and saves them to files.
    """
    try:
        # 1. Epoch-level plots
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(all_epoch_rewards, label='Total Reward')
        plt.title('Total Reward per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(all_epoch_quant_counts, label='Quant Count', color='orange')
        plt.title('Quantized Layers Count')
        plt.xlabel('Epoch')
        plt.ylabel('Count')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(all_epoch_avg_mses, label='Avg MSE', color='green')
        plt.title('Average MSE per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"plots/training_progress_epoch_{ep+1}.png")
        plt.close()

        # 2. Step-level plots (for the current epoch)
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(step_mses, label='MSE')
        plt.title(f'MSE per Step (Epoch {ep+1})')
        plt.xlabel('Layer Step')
        plt.ylabel('MSE')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(step_quant_counts, label='Cumulative Quant Count', color='orange')
        plt.title(f'Quantization Count (Epoch {ep+1})')
        plt.xlabel('Layer Step')
        plt.ylabel('Count')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(step_rmse_scales, label='RMSE/Scale', color='red')
        plt.title(f'RMSE/Scale per Step (Epoch {ep+1})')
        plt.xlabel('Layer Step')
        plt.ylabel('RMSE/Scale')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"plots/step_details_epoch_{ep+1}.png")
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting graphs: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train PPO Agent for TFLite Quantization')
    parser.add_argument('--checkpoint', '-c', type=str, help='Path to checkpoint file to load weights from')
    args = parser.parse_args()

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    setup_logging(f"train_execution_{date_str}.log")
    logging.info("Initializing Mixed-Precision Quantization via PPO...")

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
    dummy_state = tf.zeros((1, 9))
    agent(dummy_state)

    # Load Checkpoint if provided
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            logging.info(f"Loading weights from checkpoint: {args.checkpoint}")
            try:
                agent.load_weights(args.checkpoint)
                logging.info("Weights loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load weights: {e}")
        else:
            logging.error(f"Checkpoint file not found: {args.checkpoint}")
            return

    # Tracking Metrics
    episode_rewards = []
    
    # Plotting Data
    os.makedirs("plots", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    all_epoch_rewards = []
    all_epoch_quant_counts = []
    all_epoch_avg_mses = []
    
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
        
        # Step-level metrics for this epoch
        step_mses = []
        step_quant_counts = []
        step_rmse_scales = []
        
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
            
            # Collect Step Metrics
            # Note: env.current_layer_idx is already incremented in step()
            last_layer_idx = env.current_layer_idx - 1
            if last_layer_idx in env.debugger_stats:
                stat = env.debugger_stats[last_layer_idx]
                step_mses.append(stat.get('output_mse_mean', 0.0))
                step_rmse_scales.append(stat.get('rmse/scale', 0.0))
            else:
                step_mses.append(0.0)
                step_rmse_scales.append(0.0)
            step_quant_counts.append(sum(env.decision_history))
            
            # 3. Store Experience
            # PPO expects: (s, a, r, s_prime, log_prob_a, done)
            # Store unclipped 'a' to maintain Gaussian consistency in PPO update
            agent.put_data((state, a_scalar, reward, next_state, log_prob_a_scalar, done))
            
            state = next_state
            ep_reward += reward
            
        # 4. Train Agent at end of episode
        agent.train_net()
        
        episode_rewards.append(ep_reward)
        
        # Update Epoch Metrics
        all_epoch_rewards.append(ep_reward)
        all_epoch_quant_counts.append(sum(env.decision_history))
        all_epoch_avg_mses.append(np.mean(step_mses) if step_mses else 0.0)
        
        logging.info(f"Episode {ep+1}/{EPOCHS} | Total Reward: {ep_reward:.4f}")
        
        # Save weights
        try:
            agent.save_weights(f"checkpoints/ppo_agent_epoch_{ep+1}.weights.h5")
        except Exception as e:
            logging.error(f"Failed to save checkpoint/model: {e}")

        # Export TFLite model
        model_export_path = f"checkpoints/quant_model_epoch_{ep+1}.tflite"
        env.export_model(model_export_path)


        plot_training_results(ep, EPOCHS, all_epoch_rewards, all_epoch_quant_counts, all_epoch_avg_mses, step_mses, step_quant_counts, step_rmse_scales)

    logging.info("Training Complete.")

if __name__ == "__main__":
    main()
