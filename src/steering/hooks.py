import torch
import torch.nn as nn
from .pid_controller import LayerPIDController
from .angular_rotator import adaptive_angular_steering

class ClosedLoopSteeringEngine:
    def __init__(self, model: nn.Module, b1: torch.Tensor, b2: torch.Tensor, pid_params: dict, layer_range: list):
        self.model = model
        self.b1 = b1.to(model.device)
        self.b2 = b2.to(model.device)
        self.pid = LayerPIDController(**pid_params)
        self.layer_range = layer_range
        self.handles = []

    def _get_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output

            # 1. Compute dynamic angle via PID controller
            theta = self.pid.compute_theta(h, self.b1)

            # 2. Execute Adaptive Angular Steering
            h_steered = adaptive_angular_steering(h, self.b1, self.b2, theta)

            if isinstance(output, tuple):
                return (h_steered,) + output[1:]
            return h_steered
        return hook

    def register(self):
        self.pid.reset()
        for idx in self.layer_range:
            layer = self.model.model.layers[idx]
            self.handles.append(layer.register_forward_hook(self._get_hook(idx)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pid.reset()
