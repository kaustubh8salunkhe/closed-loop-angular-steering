import torch
import torch.nn as nn
from typing import Dict
from .pid_controller import LayerPIDController
from .angular_rotator import rotate_in_plane

class ClosedLoopSteeringEngine:
    def __init__(
        self,
        model: nn.Module,
        nuisance_vectors: Dict[int, torch.Tensor],
        pca_matrices: Dict[int, torch.Tensor],
        layer_range: range,
        pid_params: dict
    ):
        self.model = model
        self.nuisance_vectors = {k: v.to(model.device) for k, v in nuisance_vectors.items()}
        self.pca_matrices = {k: v.to(model.device) for k, v in pca_matrices.items()}
        self.layer_range = layer_range
        self.pid_controller = LayerPIDController(**pid_params)
        self.hook_handles = []

    def _create_hook(self, layer_idx: int):
        def forward_hook(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # 1. Retrieve layer-specific basis
            v_r = self.nuisance_vectors[layer_idx]
            w_pca = self.pca_matrices[layer_idx]  # [k, D]
            
            # 2. Compression Bottleneck: Remove spurious high-frequency directions
            # h_comp: [B, S, D] reconstructed through low-rank subspace
            h_comp = torch.matmul(torch.matmul(hidden_states, w_pca.T), w_pca)
            
            # 3. Dynamic PID Angle Computation based on compressed activation
            theta = self.pid_controller.compute_steering_angle(h_comp, v_r)
            
            # 4. Norm-Preserving 2D Rotation applied to ORIGINAL hidden states
            steered_states = rotate_in_plane(hidden_states, v_r, theta)
            
            if isinstance(output, tuple):
                return (steered_states,) + output[1:]
            return steered_states
        return forward_hook

    def register(self):
        self.pid_controller.reset_state()
        for idx in self.layer_range:
            layer = self.model.model.layers[idx]
            handle = layer.register_forward_hook(self._create_hook(idx))
            self.hook_handles.append(handle)

    def remove(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.pid_controller.reset_state()
