import os
# Suppress TensorFlow logs (Must be before importing tensorflow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import tensorflow_probability as tfp
from environment import TFLiteQuantizationEnv
from ppo import PPO
from utils import BATCH_SIZE, EPOCHS

tfd = tfp.distributions

def representative_dataset_gen():
    """
    Generator for TFLite calibration.
    Loads real calibration data from .npy files.
    """
    # Paths to calibration data
    base_dir = "/root/workspace/rt_project/model/whisper_decoder_tiny"
    
    # Load data (Assuming Decoder takes Tokens and Encoder Output)
    # Adjust filenames based on actual model requirements.
    # For 'base' model, hidden dim is 512.
    enc_output_path = os.path.join(base_dir, "calibration_encoder_output_30x1500x512.npy")
    tokens_path = os.path.join(base_dir, "calibration_random_30x224_int.npy")
    
    if os.path.exists(enc_output_path) and os.path.exists(tokens_path):
        enc_output = np.load(enc_output_path).astype(np.float32)
        tokens = np.load(tokens_path).astype(np.int32) # Tokens are usually int
        
        # Ensure we have enough data
        num_samples = enc_output.shape[0]
        
        for i in range(min(num_samples, BATCH_SIZE)):
            # Yield [tokens, enc_output] assuming this order
            yield [tokens[i:i+1], enc_output[i:i+1]]
    else:
        # Fallback to random if files missing
        print("Calibration files not found. Using random data.")
        for _ in range(BATCH_SIZE):
            # Dummy shapes for Whisper Decoder Base
            tokens = np.random.randint(0, 10000, (1, 224), dtype=np.int32)
            enc_out = np.random.randn(1, 1500, 512).astype(np.float32)
            yield [tokens, enc_out]

def main():
    print("Initializing Mixed-Precision Quantization via PPO...")
    
    # Path to the SavedModel directory
    model_path = "/root/workspace/rt_project/model/whisper_decoder_base"
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found.")
        return

    # Initialize Environment
    env = TFLiteQuantizationEnv(
        model_path=model_path, 
        representative_dataset_gen=representative_dataset_gen
    )
    
    # Initialize Agent
    agent = PPO() 
    
    # Tracking Metrics
    episode_rewards = []
    
    for ep in range(EPOCHS):
        state = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            # 1. prepare state
            state = env.prepare_state(state)

            # 2. Get Action
            mu, sigma = agent.pi(tf.convert_to_tensor([state], dtype=tf.float32))
            dist = tfd.Normal(loc=mu, scale=sigma + 1e-5)
            action = dist.sample()
            action = tf.clip_by_value(action, 0.0, 1.0)
            log_prob = dist.log_prob(action)
            
            a = action.numpy()[0][0]
            prob_a = log_prob.numpy()[0][0]
            
            # 3. Step Environment
            next_state, reward, done, info = env.step(a)
            
            # 4. Store Experience
            agent.put_data((state, a, reward, next_state, prob_a, done))
            
            state = next_state
            ep_reward += reward
            
        # 4. Train Agent at end of episode
        agent.train_net()
        
        episode_rewards.append(ep_reward)
        avg_reward = np.mean(episode_rewards[-10:])
        
        print(f"Episode {ep+1}/{EPOCHS} | Total Reward: {ep_reward:.2f} | Avg Reward (Last 10): {avg_reward:.2f}")
        
    print("Training Complete. Optimal quantization policy learned.")
    
    # Inference / Final Conversion
    print("Generating optimal TFLite model...")
    state = env.reset()
    done = False
    while not done:
        mu, sigma = agent.pi(tf.convert_to_tensor([state], dtype=tf.float32))
        action = mu # Use mean for deterministic inference
        a = action.numpy()[0][0]
        state, _, done, _ = env.step(a)
    
    final_decisions = env.decision_history
    denylisted_nodes = []
    for i, is_quantized in enumerate(final_decisions):
        if not is_quantized:
            # Map layer index i to op index
            layer = env.sorted_layers[i]
            op_idx = layer['op_idx']
            if op_idx in env.op_idx_to_name:
                denylisted_nodes.append(env.op_idx_to_name[op_idx])
                
    print("Optimal Denylisted Nodes (Float):", denylisted_nodes)

if __name__ == "__main__":
    main()
