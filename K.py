# 全部重新写过吧，我服了
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Optional, Callable, Union
from xuance.torch import Tensor, Module
from xuance.torch.utils import ModuleType, mlp_block, gru_block, lstm_block
from xuance.torch.utils import CategoricalDistribution, DiagGaussianDistribution, ActivatedDiagGaussianDistribution
from gym.spaces import Space, Discrete
from typing import Optional, Sequence, Tuple, Type, Union, Callable
import math
import numpy as np
from xuance.torch.policies.deterministic import BasicQnetwork
from copy import deepcopy


def kan_block(input_dim: int,
              output_dim: int,
              size: dict = None,
              normalize: Optional[Union[nn.BatchNorm1d, nn.LayerNorm]] = None,
              activation: Optional[ModuleType] = None,
              initialize: Optional[Callable[[
                  torch.Tensor], torch.Tensor]] = None,
              device: Optional[Union[str, int, torch.device]] = None) -> Tuple[Sequence[ModuleType], Tuple[int]]:
    block = []
    kanLinear = KANLinear(input_dim, output_dim, device=device,
                          grid_size=size["grid_size"], spline_order=size["spline_order"])
    block.append(kanLinear)
    # if activation is not None:
    # block.append(activation())
    return block, (output_dim,)


class KBasicQhead(Module):
    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 hidden_sizes: Sequence[int],
                 size: dict,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(KBasicQhead, self).__init__()
        layers_ = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            kan, input_shape = kan_block(
                input_shape[0], h, size, normalize, activation, initialize, device)
            layers_.extend(kan)
        layers_.extend(
            kan_block(input_shape[0], n_actions, size, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers_)

    def forward(self, x: Tensor):
        """
        Returns the output of the Q network.
        Parameters:
            x (Tensor): The input tensor.
        """
        return self.model(x)


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        device,
        grid_size=5+5,
        spline_order=3+5,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.device = device
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features).to(device))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
            .to(device))
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1,
                               self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        ).to(self.device)  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        x = self.process_input(x, self.device)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        # (in_features, batch_size, out_features)
        B = y.transpose(0, 1).to(self.device)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1).to(self.device)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x):
        x = self.process_input(x, self.device)
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    def process_input(self, x, device):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(device)
        elif isinstance(x, torch.Tensor):
            x = x.to(device)
        else:
            raise ValueError("Unsupported input type")
        return x


class Basic_KAN(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(Basic_KAN, self).__init__()
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for h in self.hidden_sizes:
            kan, input_shape = kan_block(input_shape[0], h, self.normalize, self.activation, self.initialize,
                                         device=self.device)
            layers.extend(kan)
        return nn.Sequential(*layers)

    def forward(self, observations: np.ndarray):
        tensor_observation = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device)
        return {'state': self.model(tensor_observation)}


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        device,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-5, 5],
    ):
        super(KAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.device = device
        self.layers = torch.nn.ModuleList()
        self.layers_hidden = layers_hidden
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    device,
                    grid_size=self.grid_size,
                    spline_order=self.spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
# 这里需要做一些改动，输入的x一般是np类型就好

    def forward(self, x: np.ndarray, update_grid=False):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return {'state': x}

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(
                regularize_activation, regularize_entropy)
            for layer in self.layers
        )


ns = argparse.Namespace()
ns.variable1 = 10
ns.variable2 = 20


class KBasicQnetwork(BasicQnetwork):
    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 size: dict,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        # 用于跳过父类的representation的deepcopy的报错
        ns = argparse.Namespace()
        ns.output_shapes = {'state': [0]}
        super(KBasicQnetwork, self).__init__(action_space,
                                             ns,
                                             [],
                                             None,
                                             None,
                                             None,
                                             device
                                             )
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = type(representation)(representation.layers_hidden, representation.device,
                                                          representation.grid_size, representation.spline_order,)
        # 这里复制参数
        for tp, ep in zip(self.target_representation.parameters(), self.representation.parameters()):
            tp.data.copy_(ep.data)
        # self.target_representation.load_state_dict(representation.state_dict())
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = KBasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,size,
                                      normalize, initialize, activation, device)
        self.target_Qhead = KBasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,size,
                                        normalize, initialize, activation, device)
        # 这里复制参数
        for tp, ep in zip(self.target_Qhead.parameters(), self.eval_Qhead.parameters()):
            tp.data.copy_(ep.data)
    # def copy_target(self):
    #     #soft_update
    #     interpolation_parameter = 1e-3
    #     for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
    #         tp.data.copy_(interpolation_parameter*ep+(1-interpolation_parameter)*tp)
    #     for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
    #         tp.data.copy_(interpolation_parameter*ep+(1-interpolation_parameter)*tp)
