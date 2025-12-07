# TFLite Mixed-Precision Quantization with PPO

This project implements a Reinforcement Learning (RL) based approach to find the optimal mixed-precision quantization policy for TensorFlow Lite (TFLite) models. It uses Proximal Policy Optimization (PPO) to balance the trade-off between model compression (quantization) and accuracy preservation.

## Features

*   **Automated Quantization Policy**: Learns which layers to quantize and which to keep in float32 to maximize compression while minimizing accuracy loss.
*   **Custom RL Environment**: Interacts directly with the TFLite Interpreter and Quantization Debugger to measure layer-wise sensitivity and MSE.
*   **PPO Agent**: Implemented in pure TensorFlow 2.x without external RL libraries (like `tensorflow-probability` or `stable-baselines`), ensuring compatibility and ease of modification.
*   **Visualization**: Generates training plots (Reward, MSE, Quantization Count) per epoch and step.
*   **Checkpointing**: Automatically saves model weights and exports the best TFLite model at each epoch.
*   **Resume Training**: Supports loading pretrained weights to continue training.

## Requirements

*   Python 3.11.x

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Configuration
Modify `src/parameters.py` to set hyperparameters, model paths, and reward weights.
*   `MODEL_PATH`: Path to the SavedModel directory.
*   `CALB_PATH`: Path to calibration data (`.npy`).
*   `W_COMPRESSION`: Weight for compression reward.
*   `W_ACCURACY`: Weight for accuracy penalty.
*   `TARGET_MSE_THRESHOLD`: Target MSE threshold.

### 2. Training
Run the training script:
```bash
python src/train.py
```
This will start the training process, logging progress to `train_execution_DATE.log` and saving outputs to `plots/` and `checkpoints/`.

### 3. Resume Training
To resume training from a specific checkpoint:
```bash
python src/train.py --checkpoint checkpoints/ppo_agent_epoch_50.weights.h5
```

## Project Structure

*   `src/train.py`: Main entry point for training the PPO agent.
*   `src/environment.py`: Defines `TFLiteQuantizationEnv`, handling TFLite interaction and reward calculation.
*   `src/ppo.py`: Implementation of the PPO agent (Actor-Critic network).
*   `src/parameters.py`: Hyperparameters and configuration settings.
*   `src/utils.py`: Utility functions for logging and model analysis.
*   `src/hooks.py`: Monkey patches for TFLite interpreter to enable detailed debugging.
*   `model/`: Directory containing target models and calibration data.
*   `plots/`: Generated training graphs.
*   `checkpoints/`: Saved model weights and exported TFLite models.
*   `log/`: Execution logs.

## How it Works

1.  **State**: The agent observes the current layer's statistics (input range, variance, size) and the accumulated sensitivity of the model.
2.  **Action**: The agent outputs a continuous "sensitivity" score. If the score > threshold, the layer is quantized; otherwise, it remains float.
3.  **Reward**: The agent receives a positive reward for quantizing layers (compression) and a negative penalty if the quantization increases the Mean Squared Error (MSE) beyond a target threshold.
4.  **Optimization**: The PPO algorithm updates the policy to maximize the expected cumulative reward.

## Outputs

*   **Logs**: Detailed execution logs in `log/`.
*   **Plots**:
    *   `training_progress_epoch_N.png`: Total Reward, Quantization Count, Avg MSE per epoch.
    *   `step_details_epoch_N.png`: Layer-wise MSE and decisions for a specific epoch.
*   **Models**:
    *   `checkpoints/quant_model_epoch_N.tflite`: The TFLite model quantized according to the policy at epoch N.
