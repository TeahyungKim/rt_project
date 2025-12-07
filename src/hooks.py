import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

import numpy as np

from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export

import tensorflow as tf

def __hook_init__(self : tf.lite.experimental.QuantizationDebugOptions,
            layer_debug_metrics: Optional[Mapping[str,
                                                Callable[[np.ndarray],
                                                            float]]] = None,
            model_debug_metrics: Optional[Mapping[
                str, Callable[[Sequence[np.ndarray], Sequence[np.ndarray]],
                            float]]] = None,
            layer_direct_compare_metrics: Optional[Mapping[str, Callable[
                # [HOOK HERE]..
                [Sequence[np.ndarray], Sequence[np.ndarray], float, int,
                Optional[_interpreter.Interpreter], Optional[List[str]], Optional[Tuple[int, int]]],
                float]]] = None,
            denylisted_ops: Optional[List[str]] = None,
            denylisted_nodes: Optional[List[str]] = None,
            fully_quantize: bool = False) -> None:
    self.layer_debug_metrics = layer_debug_metrics
    self.model_debug_metrics = model_debug_metrics
    self.layer_direct_compare_metrics = layer_direct_compare_metrics
    keys = []
    for metrics in [
        layer_debug_metrics, model_debug_metrics, layer_direct_compare_metrics
    ]:
        if metrics is not None:
            keys.extend(metrics.keys())
    if len(keys) != len(set(keys)):
        raise ValueError('Provided metrics have duplicate keys.')
    self.denylisted_ops = denylisted_ops
    self.denylisted_nodes = denylisted_nodes
    self.fully_quantize = fully_quantize

@tf_export.tf_export("lite.experimental.QuantizationDebugger")
def _hook_collect_layer_statistics(self : tf.lite.experimental.QuantizationDebugger) -> Dict[str, Dict[str, float]]:
    layer_statistics = collections.defaultdict(
        lambda: collections.defaultdict(list))

    initialize = True
    for tensor_data in self._data_gen():
        self._set_input_tensors(self._quant_interpreter, tensor_data, initialize)
        initialize = False

        # Run the model.
        self._quant_interpreter.invoke()

        # Collect the statistics of this invoke result.
        for tensor_detail in self._get_numeric_verify_tensor_details():
            tensor_name = tensor_detail['name']  # pytype: disable=unsupported-operands  # dynamic-method-lookup
            diffs = self._quant_interpreter.get_tensor(tensor_detail['index'])  # pytype: disable=unsupported-operands  # dynamic-method-lookup
            for metric_name, metric_fn in self._layer_debug_metrics.items():
                layer_statistics[tensor_name][metric_name].append(metric_fn(diffs))

        if self._debug_options.layer_direct_compare_metrics is not None:
            for i, tensor_detail in enumerate(self._get_numeric_verify_tensor_details()):
                tensor_name = tensor_detail['name']  # pytype: disable=unsupported-operands  # dynamic-method-lookup
                op_idx = self._defining_op[tensor_detail['index']]  # pytype: disable=unsupported-operands  # dynamic-method-lookup
                op_detail = self._quant_interpreter._get_op_details(op_idx)  # pylint: disable=protected-access
                q_idx, f_idx = op_detail['inputs']
                quant_input_detail = self._quant_interpreter._get_tensor_details(  # pylint: disable=protected-access
                    q_idx, subgraph_index=0)
                for (metric_name, metric_fn
                    ) in self._debug_options.layer_direct_compare_metrics.items():
                    layer_statistics[tensor_name][metric_name].append(
                        metric_fn(
                            self._quant_interpreter.get_tensor(f_idx),
                            self._quant_interpreter.get_tensor(q_idx),
                            quant_input_detail['quantization_parameters']['scales'][0],
                            quant_input_detail['quantization_parameters']['zero_points'][0],
                            # [HOOK HERE]..
                            self._quant_interpreter, op_detail,
                            (i, len(self._get_numeric_verify_tensor_details()))
                            ))
    # Calculate final aggregated metrics for each layer.
    for metrics in layer_statistics.values():
        for metric_name in metrics:
            metrics[metric_name] = np.nanmean(metrics[metric_name])

    return layer_statistics

class TfHooks:


    @staticmethod
    def tf_hook_init():
        # HACK: Modify Interpreter to always preserve tensors for debugging
        # Check if already patched to prevent recursive wrapping
        if not getattr(_interpreter.Interpreter, "_is_patched", False):
            original_init = _interpreter.Interpreter.__init__
            def new_init(self, *args, **kwargs):
                # Force the option to True, overriding any provided value
                kwargs['experimental_preserve_all_tensors'] = True
                original_init(self, *args, **kwargs)            
            _interpreter.Interpreter.__init__ = new_init
            _interpreter.Interpreter._is_patched = True

        tf.lite.experimental.QuantizationDebugOptions.__init__ = __hook_init__
        tf.lite.experimental.QuantizationDebugger._collect_layer_statistics = _hook_collect_layer_statistics

