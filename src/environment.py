import os
import io
import logging

# Suppress TensorFlow logs (Must be before importing tensorflow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import pandas as pd

from typing import List, Dict, Tuple
from utils import ModelUtils
from hooks import TfHooks
from parameters import QUANTIZATION_THRESHOLD, ENABLE_DEBUG, W_COMPRESSION, W_ACCURACY, TARGET_MSE_THRESHOLD

tf.get_logger().setLevel(logging.ERROR)

class TFLiteQuantizationEnv:
    """
    Reinforcement Learning Environment for TFLite Mixed-Precision Quantization.
    """
    def __init__(self, model_path: str, calib_idx: int, calib_data_list: List[np.ndarray]):
        TfHooks.tf_hook_init()

        # tf.lite.experimental.QuantizationDebugOptions.__init__ = __hook_init__
        # tf.lite.experimental.QuantizationDebugger._collect_layer_statistics = _hook_collect_layer_statistics
        
        self.model_path = model_path
        
        # Ensure model_path is a directory (SavedModel)
        if not os.path.isdir(self.model_path):
            raise ValueError("model_path must be a directory containing a SavedModel.")

        # Initial conversion to analyze structure
        self.converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
        self.f32_tflite_model = self.converter.convert()
        self.sorted_layers, self.tensor_details, self.input_details, self.output_details, \
        self.tensor_outputs, self.output_tensor_histories, self.output_tensor_targets = \
            ModelUtils.get_model_details(self.f32_tflite_model, calib_idx, calib_data_list)
        self.n_layers = len(self.sorted_layers)

        logging.info(f"Model({calib_idx}) loaded with {self.n_layers} layers for quantization.")

        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def debug_dataset_gen():
            yield_data_dict = []
            for i, input_detail in enumerate(self.input_details):
                calb_data = calib_data_list[i]                    
                calb_data = calb_data[calib_idx:calib_idx+1]
                yield_data_dict.append(tf.cast(
                    tf.convert_to_tensor(calb_data), calb_data.dtype
                ))
            yield yield_data_dict

        self.debug_dataset_gen = debug_dataset_gen
        self.converter.representative_dataset = self.debug_dataset_gen

        # Initialize state variables
        self.current_layer_idx = 0
        self.decision_history = []  # True=Quantize, False=Float
        self.sensitivity_history = []
        self.cumulative_sensitivity_weighted = 0.0
        self.debugger_stats = {}

    def summary(self):
        """
        Prints model summary.
        """
        # Print topological order
        logging.info("Layer inputs:")
        for input_detail in self.input_details:
            logging.info(f"Model Input Tensor: Index[{input_detail['index']}] Name[{input_detail['name']}] Shape{input_detail['shape']} DType[{input_detail['dtype']}]")

        logging.info("Topological Order of Layers:")
        for i, layer in enumerate(self.sorted_layers):
            op_info = layer['op_info']
            output_idx = layer['output_idx']
            logging.info(f" - Layer{i}: [{op_info['op_name']}] Output[{output_idx}] <- Inputs[{op_info['inputs']}]")
            logging.info(f" ------------[{self.tensor_outputs[output_idx]}]")

        logging.info("Static Constant Tensors:")
        for i, (name, stats) in enumerate(self.output_tensor_histories.items()):
            logging.info(f" - Tensor{i}: [{name}] Stats: {stats}")

    def reset(self) -> np.ndarray:
        """
        Resets the environment.
        """
        self.current_layer_idx = 0
        self.target_quant_layers = []
        self.decision_history = []
        self.sensitivity_history = []
        self.cumulative_sensitivity_weighted = 0.0
        self.debugger_stats = {}
 
        # Run debugger for the first layer in advance
        self._run_debugger(target_layer_idx=0)
        return self._construct_next_state(0)
 
 
    def step(self, action_sensitivity: float) -> Tuple[np.ndarray, float, bool]:
        """
        Executes one step.
        """
        # Interpret action
        is_quantized = action_sensitivity > QUANTIZATION_THRESHOLD
        self.target_quant_layers.append(self.current_layer_idx) if is_quantized else None
        
        # Update history
        self.decision_history.append(is_quantized)
        self.sensitivity_history.append(action_sensitivity)
        self.cumulative_sensitivity_weighted += is_quantized * action_sensitivity
        
        # Calculate reward
        reward = self._calculate_reward(self.current_layer_idx, is_quantized)

        # Move to next layer
        next_layer_idx = self.current_layer_idx + 1
        done = next_layer_idx >= self.n_layers
        if not done:
            self._run_debugger(next_layer_idx)

        next_state = self._construct_next_state(next_layer_idx, done)
        if ENABLE_DEBUG:
            logging.debug(f"Step: Layer {self.current_layer_idx}, Action(Sensitivity)={action_sensitivity:.4f}, Quantized={is_quantized}, Reward={reward:.4f}, Done={done}")
        self.current_layer_idx = next_layer_idx
        return next_state, reward, done

    def export_model(self, output_path: str):
        """
        Exports the final quantized model based on the current decision history.
        """
        temp_target_quant_layers = set(self.target_quant_layers)
        denylisted_nodes = []
        for idx, layer in enumerate(self.sorted_layers):
            if idx not in temp_target_quant_layers:
                denylisted_nodes.append(layer['output_name'])
        
        logging.info(f"Exporting model to {output_path}")
        logging.info(f"Total Layers: {len(self.sorted_layers)}, Quantized: {sum(self.decision_history)}, Float: {len(denylisted_nodes)}")

        debug_options = tf.lite.experimental.QuantizationDebugOptions(
            denylisted_ops=['SPLIT'],
            denylisted_nodes=denylisted_nodes
        )

        debugger = tf.lite.experimental.QuantizationDebugger(
            converter=self.converter,
            debug_dataset=self.debug_dataset_gen,
            debug_options=debug_options
        )
        debugger.run()

        with open(output_path, 'wb') as f:
            f.write(debugger.get_nondebug_quantized_model())

    def _run_debugger(self, target_layer_idx: int):
        """
        Runs the QuantizationDebugger.
        Quantizes target layers and keeps others as float.
        """
        temp_target_quant_layers = set(self.target_quant_layers)
        temp_target_quant_layers.add(target_layer_idx)
        denylisted_nodes = []
        target_quant_layer_name = None
        
        for idx, layer in enumerate(self.sorted_layers):
            if idx == target_layer_idx:
                target_quant_layer_name = layer['output_name']
            if idx not in temp_target_quant_layers:
                denylisted_nodes.append(layer['output_name'])
        
        if ENABLE_DEBUG:
            logging.info(f"Target Layer_{target_layer_idx} : [{target_quant_layer_name}]")
            logging.info(f"Quantized Layers: [{len(temp_target_quant_layers)}]")
        
        # Configure debugger options
        captured_values = {
            'name': None,
            'modified_name': None,            
            'input_stats': [],
            'output_tensor_targets': []
        }
        
        def capture_metrics(f, q, s, zp, interpreter, verify_op_detail, iter_info):
            # Capture metrics for the last iteration
            if (iter_info[0] + 1) < iter_info[1]:
                return 0

            q_output_idx, f_output_idx = verify_op_detail['inputs']
            f_output_detail = interpreter._get_tensor_details(f_output_idx, subgraph_index=0)
            q_output_detail = interpreter._get_tensor_details(q_output_idx, subgraph_index=0)
            
            if target_quant_layer_name != f_output_detail['name']:
                logging.warning(f"Mismatched target layer in capture_metrics."\
                    f" [{target_quant_layer_name}] != [{f_output_detail['name']}], qname: {q_output_detail['name']}")
            captured_values['name'] = target_quant_layer_name
            captured_values['modified_name'] = f_output_detail['name']
      
            def find_op_info(interpreter, idx):
                ops_details = interpreter._get_ops_details()
                for op_info in ops_details:
                    if idx in op_info['outputs']:
                        return op_info
                return None

            op_info = find_op_info(interpreter, f_output_idx)
            if ENABLE_DEBUG:
                logging.debug(f"[{iter_info}] Found Op Info for target layer: {op_info['op_name']}, inputs: {op_info['inputs']}, outputs: {op_info['outputs']}")

            self.output_tensor_histories[target_quant_layer_name] = ModelUtils.make_history(f)
            op_inputs = op_info['inputs']
            
            for i, in_idx in enumerate(op_inputs):
                in_tensor_detail = interpreter._get_tensor_details(in_idx, subgraph_index=0)
                if ENABLE_DEBUG:
                    logging.debug(f"--[{i}/{len(op_inputs)}] tensor({in_idx}) shape: {in_tensor_detail['shape']} name: {in_tensor_detail['name']}, index: {in_tensor_detail['index']}")
                if in_tensor_detail['name'] not in self.output_tensor_histories:
                    # Handle constants or model inputs
                    in_tensor_data = interpreter.tensor(in_idx)()
                    history = ModelUtils.make_history(in_tensor_data)
                else:
                    history = self.output_tensor_histories[in_tensor_detail['name']]
                captured_values['input_stats'].append(history)

            output_details = interpreter.get_output_details()
            if not captured_values['output_tensor_targets']:
                captured_values['output_tensor_targets'] = [[] for _ in output_details]
            for i, out_detail in enumerate(output_details):
                captured_values['output_tensor_targets'][i].append(interpreter.get_tensor(out_detail['index']))

            return 0

        debug_options = tf.lite.experimental.QuantizationDebugOptions(
            denylisted_ops=['SPLIT'],
            denylisted_nodes=denylisted_nodes,
            layer_direct_compare_metrics={
                'target_name': capture_metrics
            }
        )

        debugger = tf.lite.experimental.QuantizationDebugger(
            converter=self.converter,
            debug_dataset=self.debug_dataset_gen,
            debug_options=debug_options
        )
        debugger.run()

        string_io = io.StringIO()
        debugger.layer_statistics_dump(string_io)
        string_io.seek(0)
        stats_df = pd.read_csv(string_io)

        # Add input stats to the dataframe
        mask = stats_df['tensor_name'] == target_quant_layer_name
        if mask.any():
            idx = stats_df[mask].index[0]
            stats_df.loc[idx, 'input_count'] = len(captured_values['input_stats'])
            input_stat = ModelUtils.make_average_history(captured_values['input_stats'])
            stats_df.loc[idx, 'input_elems'] = input_stat['count']
            stats_df.loc[idx, 'input_mean'] = input_stat['mean']
            stats_df.loc[idx, 'input_std'] = input_stat['std']
            stats_df.loc[idx, 'input_min'] = input_stat['min']
            stats_df.loc[idx, 'input_max'] = input_stat['max']
            total_mse = 0.0
            for i, batchdata in enumerate(captured_values['output_tensor_targets']):
                output_tensor = ModelUtils.get_average_output_tensor_target(batchdata)
                # Calculate MSE
                target_tensor = self.output_tensor_targets[i]
                mse = np.mean((output_tensor - target_tensor) ** 2)
                tensor_name = self.output_details[i]['name']
                if ENABLE_DEBUG:
                    logging.debug(f"Output Tensor[{tensor_name}] MSE: {mse}")
                stats_df.loc[idx, f'output_mse{i}'] = mse
                total_mse += mse
            stats_df.loc[idx, f'output_count'] = len(captured_values['output_tensor_targets'])
            stats_df.loc[idx, 'output_mse_mean'] = total_mse / len(captured_values['output_tensor_targets'])
        else:
            # Handle missing target tensor
            logging.error(stats_df)
            raise Warning(f"Target tensor [{target_quant_layer_name}] not found in stats DataFrame[{stats_df['tensor_name']}].")

        def update_rmse(stats_df):
            stats_df['rmse'] = stats_df.apply(
                lambda row: np.sqrt(row['mean_squared_error']), axis=1)
            stats_df.insert(1, 'rmse', stats_df.pop('rmse'))
            stats_df['rmse/scale'] = stats_df.apply(
                lambda row: row['rmse'] / row['scale'], axis=1)
            stats_df.insert(1, 'rmse/scale', stats_df.pop('rmse/scale'))

        update_rmse(stats_df)

        # Store the last row of stats
        self.debugger_stats[target_layer_idx] = stats_df.iloc[-1]    

        if ENABLE_DEBUG:
            os.makedirs('./debug', exist_ok=True)
            stats_df.to_csv(f"./debug/debugger_stats_layer_{target_layer_idx}.csv")
            with open(f'./debug/quant_model_{target_layer_idx}.tflite', 'wb') as f:
                f.write(debugger.quant_model)

    def _construct_next_state(self, next_layer_idx: int, cur_done: bool = False) -> np.ndarray:
        """
        Constructs the 9-dim state vector.
        """
        if cur_done:
            return np.zeros((9,), dtype=np.float32)
        layer_info = self.sorted_layers[next_layer_idx]
        
        # Normalize features to avoid saturation
        n_layers_float = float(max(1, self.n_layers))
        
        s1 = float(next_layer_idx) / n_layers_float
        s2 = float(hash(layer_info['op_name']) % 1000) / 1000.0
        
        stat = self.debugger_stats[next_layer_idx]
        
        # Log scale for counts to handle large variance
        s3 = np.log1p(float(stat['input_count'])) 
        s4 = np.log1p(float(stat['input_elems'])) / 10.0 
        
        # Tanh for statistics to bound them [-1, 1]
        s5 = np.tanh(float(stat['input_mean']))
        s6 = np.tanh(float(stat['input_std']))
        
        # Handle potential infinity/NaN in rmse/scale
        val_s7 = float(stat['rmse/scale'])
        if np.isfinite(val_s7):
            s7 = np.tanh(val_s7)
        else:
            s7 = 1.0
            
        s8 = self.cumulative_sensitivity_weighted / n_layers_float
        s9 = float(sum(self.decision_history)) / n_layers_float
        
        state = np.array([s1, s2, s3, s4, s5, s6, s7, s8, s9], dtype=np.float32)
        return np.nan_to_num(state)

    def _calculate_reward(self, layer_idx: int, is_quantized: bool) -> float:
        """
        Calculates reward based on quantization bonus and accuracy penalty.
        """
        # Reward for quantization
        r_quant = 1.0 if is_quantized else 0.0

        # Accuracy penalty based on MSE
        stat = self.debugger_stats[layer_idx]
        avg_mse = stat['output_mse_mean']
        r_acc = np.tanh(avg_mse / TARGET_MSE_THRESHOLD)

        # Combine rewards
        total_reward = (W_COMPRESSION * r_quant) - (W_ACCURACY * r_acc)
        if ENABLE_DEBUG:
            logging.debug(f"Layer {layer_idx}: Quant={r_quant}, MSE={avg_mse:.6f}, R_Acc={r_acc:.4f}, Total={total_reward:.4f}")
        return total_reward
