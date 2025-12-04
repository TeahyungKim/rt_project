import os
import numpy as np
import tensorflow as tf
import logging
from typing import Dict

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

class ModelUtils:
    @staticmethod
    def make_average_history(histories) -> Dict[str, float]:
        """
        Computes average statistics from a list of history dictionaries.
        """
        if not histories:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        avg_history = {
            'count': np.mean([h['count'] for h in histories]),
            'mean': np.mean([h['mean'] for h in histories]),
            'std': np.mean([h['std'] for h in histories]),
            'min': np.mean([h['min'] for h in histories]),
            'max': np.mean([h['max'] for h in histories])
        }
        return avg_history

    @staticmethod
    def make_history(tensor: np.ndarray) -> Dict[str, float]:
        """
        Computes statistics for a given tensor.
        """
        if tensor is None or tensor.size == 0:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        else:
            return {
                'count': tensor.size,
                'mean': float(np.mean(tensor)),
                'std': float(np.std(tensor)),
                'min': float(np.min(tensor)),
                'max': float(np.max(tensor))
            }

    @staticmethod
    def get_average_output_tensor_target(data_list):
        """
        Computes average output tensor from a list of data.
        """
        stacked = np.array(data_list)
        return np.mean(stacked, axis=0)
    
    @staticmethod
    def get_model_details(model_content, calib_idx, calb_data_list):
        """
        Analyzes TFLite model to extract topological order and tensor details.
        """
        interpreter = tf.lite.Interpreter(model_content=model_content,
              experimental_preserve_all_tensors=True)
        interpreter.allocate_tensors()
        
        # Access private attribute to get graph structure
        ops_details = interpreter._get_ops_details()
        tensor_details = interpreter.get_tensor_details()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Build Graph for Topological Sort
        tensor_to_producer = {}
        op_children = {i: [] for i in range(len(ops_details))}
        op_in_degree = {i: 0 for i in range(len(ops_details))}

        # HACK: Exclude certain ops from sorted layers
        SKIP_OPS = ['DELEGATE']

        # 1. Map tensors to producers
        for op_idx, op_info in enumerate(ops_details):
            if op_info['op_name'] in SKIP_OPS:
                continue
            for out_idx in op_info['outputs']:
                tensor_to_producer[int(out_idx)] = op_idx

        # 2. Build Op dependency graph
        for op_idx, op_info in enumerate(ops_details):
            if op_info['op_name'] in SKIP_OPS:
                continue
            for in_idx in op_info['inputs']:
                in_idx = int(in_idx)
                if in_idx in tensor_to_producer:
                    parent_op_idx = tensor_to_producer[in_idx]
                    if op_idx not in op_children[parent_op_idx]:
                        op_children[parent_op_idx].append(op_idx)
                        op_in_degree[op_idx] += 1

        # 3. Topological Sort
        queue = [op_idx for op_idx in range(len(ops_details)) if op_in_degree[op_idx] == 0]
        sorted_ops_indices = []
        
        while queue:
            curr_op_idx = queue.pop(0)
            sorted_ops_indices.append(curr_op_idx)
            
            for child_idx in op_children[curr_op_idx]:
                op_in_degree[child_idx] -= 1
                if op_in_degree[child_idx] == 0:
                    queue.append(child_idx)
        
        # Create sorted layers based on output tensors
        sorted_layers = []
        output_tensor_names = {}

        # HACK: Exclude certain ops from sorted layers
        SKIP_OPS = ['DELEGATE', 'SPLIT']
        for op_idx in sorted_ops_indices:
            op_info = ops_details[op_idx]
            if op_info['op_name'] in SKIP_OPS:
                continue
            # Create layer entry for each output tensor
            for out_idx in op_info['outputs']:
                sorted_layers.append({
                    'op_idx': op_idx,
                    'op_name': op_info['op_name'],
                    'output_idx': out_idx,
                    'output_name': tensor_details[out_idx]['name'],
                    'op_info': op_info
                })
                output_tensor_names[out_idx] = tensor_details[out_idx]['name']


        # 4. Extract Statistics for Constant Tensors (Weights/Biases)
        constant_tensor_stats = {}
        model_input_indices = [i['index'] for i in input_details]
        for tensor in tensor_details:
            idx = tensor['index']
            name = tensor['name']

            # Skip intermediate activations
            if idx in tensor_to_producer:
                continue
            
            # Skip model inputs
            if idx in model_input_indices:
                continue

            # Process constants (Weights/Biases)
            try:
                data = interpreter.get_tensor(idx)
                if data is not None and data.size > 0 and np.issubdtype(data.dtype, np.number):
                    constant_tensor_stats[name] = ModelUtils.make_history(data)
            except (ValueError, RuntimeError):
                continue

        # 5. Collect Input Tensor Stats using calibration data
        for i, input_detail in enumerate(input_details):
            constant_tensor_stats[input_detail['name']] = ModelUtils.make_history(calb_data_list[i])

        # 6. Collect Output Tensor Targets using calibration data
        output_tensor_targets = []
        for i, input_detail in enumerate(input_details):
            interpreter.set_tensor(input_detail['index'], calb_data_list[i][calib_idx:calib_idx+1])
        interpreter.invoke()
        output_data = []
        for output_detail in output_details:
            output_data.append(interpreter.get_tensor(output_detail['index']))
        output_tensor_targets.append(output_data)

        return sorted_layers, tensor_details, input_details, output_details, \
               output_tensor_names, constant_tensor_stats, output_tensor_targets

    @staticmethod
    def calculate_mse(arr1: np.ndarray, arr2: np.ndarray) -> float:
        """
        Calculates Mean Squared Error (MSE).
        """
        if arr1.size == 0 or arr2.size == 0:
            return 0.0
        return np.mean((arr1.flatten() - arr2.flatten()) ** 2)

    @staticmethod
    def calculate_rmse(arr1: np.ndarray, arr2: np.ndarray) -> float:
        """
        Calculates Root Mean Squared Error (RMSE).
        """
        return np.sqrt(ModelUtils.calculate_mse(arr1, arr2))
