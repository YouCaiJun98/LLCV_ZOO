import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import utils
from models.wrapper import BARStructuredWrapper
from models.unet import module_name

def exp_progress_fn(p: float, a: float = 4.) -> float:
    c = 1. - np.exp(-a)
    exp_progress = 1. - np.exp(-a * p)
    return exp_progress / c


def sigmoid_progress_fn(p: float, a: float) -> float:
    b = 1. / (1. + np.exp(a * 0.5))
    sigmoid_progress = 1. / (1. + np.exp(a * (0.5 - p)))
    sigmoid_progress = (sigmoid_progress - b) / (1. - 2. * b)
    return sigmoid_progress


class DistillationLoss(nn.Module):
    r"""
    Distillation objective.

    Args:
        T (float): The temperature.
        alpha (float): The coefficient to controll the trade-off between distillation loss and origin loss.
    """
    def __init__(self, T: float, alpha: float) -> None:
        super(DistillationLoss, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, output: Tensor, target: Tensor, label: Tensor) -> Tensor:
        r"""
        Args:
            output (Tensor): Output of the network to be trained.
            target (Tensor): Output of the teacher network.
            label (Tensor): Label of the input.

        Returns:
            Tensor: The calculated loss.
        """
        p = F.softmax(target / self.T, dim = 1)
        log_q = F.log_softmax(output / self.T, dim = 1)
        entropy = - torch.sum(p * log_q, dim = 1)
        kl = F.kl_div(log_q, p, reduction = "mean")
        loss = torch.mean(entropy + kl)
        return self.alpha * self.T ** 2 * loss + \
                F.cross_entropy(output, label) * (1 - self.alpha)

def get_previous_name(net:nn.Module, curr_name: str):
    r"""
    Get module names of the current layer's inputs.

    Args:
        net (nn.Module): The whole module.
        curr_name (str): Name of the current layer.
    Returns:
        name_list(list): Names of the current layer's inputs.
    """
    nodes = list(net.graph_module.graph.nodes)
    name_list = []
    for node in nodes:
        if node.name == curr_name:

            inputs = node.all_input_nodes
            for node in inputs:
                # we want names of convs connected ahead of the current layer.
                ruleout_names = ['relu', 'cat', 'pool']

                def undesired_name(name, undesired_names):
                    for undesired_name in undesired_names:
                        if undesired_name in name:
                            return True
                    return False

                if undesired_name(node.name, ruleout_names):
                    name_list += get_previous_name(net, node.name)
                else:
                    # this should be 'conv_xx', 'up_xx' or 'x'
                    name_list.append(node.name)
    return name_list

def get_alive_inchannels(net:nn.Module, in_layers: list, stochastic: bool):
    r"""
    Calculate the number of alive inchannels.

    Args:
        net  (nn.Module): The network to be pruned.
        in_layers (list): Name of modules connected ahead of the current layer.
        stochastic(nool): True for training and False for deterministic inference.

    Returns:
        alive_inchannels (float): Alive in channels.
    """
    alive_inchannels = 0
    for in_layer in in_layers:
        if in_layer == 'x':
            # here we assume the model is U-Net and the first layer is conv1_1
            alive_inchannels += net.conv1_1.module.in_channels
        else:
            for name, m in net.named_modules():
                if name == in_layer:
                    if stochastic:
                        alive_probability = torch.sigmoid(
                        m.log_alpha - m.beta * np.log(-m.gamma / m.zeta))
                    else:
                        alive_probability = m.cal_mask(stochastic = False)
                        alive_probability = (alive_probability > 0.).long()
                    alive_inchannels += torch.sum(alive_probability)
    return alive_inchannels

class BudgetLoss(nn.Module):
    r"""
    Original Budget loss for BAR, the budget can be chosen from ['area', 'flops'].
    """
    def __init__(self) -> Tensor:
        super(BudgetLoss, self).__init__()

    def forward(self, net: nn.Module, mode: str) -> Tensor:
        r"""
        Calculate the budget loss.

        Args:
            net (nn.Module): The network to be pruned.

        Returns:
            loss (Tensor): The budget loss.
        """
        assert mode in ['area', 'flops'], 'Can only calculate overhead in area or flops.'
        loss = torch.zeros(1).to(utils.net_device(net))
        for name, m in net.named_modules():
            if isinstance(m, BARStructuredWrapper):
                # Probability of being alive for each feature map
                alive_probability = torch.sigmoid(
                        m.log_alpha - m.beta * np.log(-m.gamma / m.zeta))
                # area loss
                module_budget_loss = torch.sum(alive_probability) * m.output_area
                if mode == 'flops':
                    alive_inchannels = 0
                    in_layers = get_previous_name(net, name)
                    alive_inchannels = get_alive_inchannels(net, in_layers, stochastic=True)
                    assert alive_inchannels, 'The computed alive inchannels is 0.'
                    # flops loss
                    module_budget_loss *= (m.module.kernel_size[0] ** 2) * alive_inchannels
                loss += module_budget_loss
        return loss

class LatencyLoss(nn.Module):
    r"""
    Latency Budget loss based on the hardware model.

    Args:
        hardware_model (nn.Module): The specified hardware model, currently it is a MLP.

    Returns:
        latency (Tensor): The predicted latency.
    """
    def __init__(self, hardware_model):
        super(LatencyLoss, self).__init__()
        self.hwm = hardware_model.eval() # TODO debug: check whether this influence the gradient of hwm. 

    def forward(self, model: nn.Module) -> Tensor:
        sparsity_ratio = torch.zeros([1, 22]).to(utils.net_device(model))
        for name, m in model.named_modules():
            if isinstance(m, BARStructuredWrapper):
                # we group gamma of every layer into a new tensor, and feed this tensor to hwm.
                alive_probability = torch.sigmoid(
                        m.log_alpha - m.beta * np.log(-m.gamma / m.zeta))
                alive_ratio = torch.sum(alive_probability) / m.module.out_channels
                sparsity_ratio[0][module_name.index(name)] = alive_ratio
        latency = torch.clip(self.hwm(sparsity_ratio)[0], 0., 1.)

        return latency

class BARStructuredLoss(nn.Module):
    r"""
    Objective of Budget-Aware Regularization Structured Pruning.

    Args:
        budget (float): The budget.
        epochs (int): Total pruning epochs.
        progress_func (str): Type of progress function ("sigmoid" or "exp"). Default: "sigmoid".
        _lambda (float): Coefficient for trade-off of sparsity loss term. Default: 1e-5.
        distillation_temperature (float): Knowledge Distillation temperature. Default: 4.
        distillation_alpha (float): Knowledge Distillation alpha. Default: 0.9.
        tolerance (float): Default: 0.01.
        margin (float): Parameter a in Eq. 5 of the paper. Default: 0.0001.
        sigmoid_a (float): Slope parameter of sigmoidal progress function. Default: 10.
        upper_bound (float): Default: 1e10.
    """
    def __init__(self, budget: float, epochs: int, progress_func: str = "sigmoid",
                 budget_mode: str = "area", _lambda: float = 1e-5, distillation_temperature: float = 4.,
                 distillation_alpha: float = 0.9, tolerance: float = 0.01, margin: float = 1e-4,
                 sigmoid_a: float = 10., upper_bound: float = 1e10, hardware_model: nn.Module=None,
                 ori_overhead: int = None) -> None:
        super(BARStructuredLoss, self).__init__()
        assert budget_mode in ['area', 'flops', 'latency'], \
            "Specified budget type {} is unsupported.".format(budget_mode)
        self.budget = budget
        self.mode = budget_mode
        self.epochs = epochs
        self.hardware_model = hardware_model

        '''
        # distillation hyperparameters
        self.distillation_alpha = distillation_alpha
        self.distillation_temperature = distillation_temperature
        '''

        self.progress_func = progress_func
        self.sigmoid_a = sigmoid_a
        self.tolerance = tolerance
        self.upper_bound = upper_bound

        self._lambda = _lambda
        self.margin = margin
        # TODO: check how to implement distill loss here.
        # self.distill_criterion = DistillationLoss(distillation_temperature, distillation_alpha)
        self.pixel_criterion = nn.L1Loss()
        self.budget_criterion = BudgetLoss() if self.mode != 'latency' else \
                                LatencyLoss(hardware_model)

        if ori_overhead:
            self._origin_overhead = ori_overhead
        else:
            self._origin_overhead = None

    def forward(self, output: Tensor, target: Tensor, 
            net: nn.Module, current_epoch_fraction: float) -> Tensor:
        r"""
        Calculate the objective.

        Args:
            Currently Removed - input (Tensor): Input image.
            output (Tensor): Output image.
            target (Tensor): Label of the image,
            net (nn.Module): The network to be updated.
            Currently Removed - teacher (nn.Module): Teacher network for distillation.
            current_epoch_fraction (float): Current epoch fraction.

        Returns:
            loss (Tensor): The loss.
        """
        # Step 1: Calculate the L1 loss.
        pixel_loss = self.pixel_criterion(output, target)

        # Step 2: Calculate the Distillation Loss.
        '''
        with torch.no_grad():
            teacher_output = teacher(input)
        classification_loss = self.classification_criterion(output, teacher_output, target)
        '''

        # Step 3: Calculate the budget loss.
        budget_loss = self.budget_criterion(net, self.mode) if self.mode != 'latency' \
            else self.budget_criterion(net)

        # Step 4: Calculate the coefficient of the budget loss.
        current_overhead = self.current_overhead(net, self.mode)
        origin_overhead = self.origin_overhead(net, self.mode)
        tolerant_overhead = (1. + self.tolerance) * origin_overhead

        p = current_epoch_fraction / self.epochs
        if self.progress_func == "sigmoid":
            p = sigmoid_progress_fn(p, self.sigmoid_a)
        elif self.progress_func == "exp":
            p = exp_progress_fn(p)

        current_budget = (1 - p) * tolerant_overhead + p * self.budget * origin_overhead

        margin = tolerant_overhead * self.margin
        lower_bound = self.budget * origin_overhead - margin
        budget_respect = (current_overhead - lower_bound) / (current_budget - lower_bound)
        budget_respect = max(budget_respect, 0.)

        if budget_respect < 1.:
            lamb_mult = min(budget_respect ** 2 / (1. - budget_respect), self.upper_bound)
        else:
            lamb_mult = self.upper_bound

        # Step 4: Combine the objectives.
        # for display 
        weighted_budget_loss = self._lambda / len(output) * lamb_mult * budget_loss
        loss = pixel_loss + weighted_budget_loss
        return loss, pixel_loss, weighted_budget_loss

    def current_overhead(self, net: nn.Module, mode:str='area') -> float:
        r"""
        Calculate the computation overhead after pruning.

        Args:
            net (nn.Module): The network to be calculated.
            mode(str): The format of overhead, could be area or flops.
        Returns:
            overhead (float): The computation overhead after pruning.
        """
        assert mode in ['area', 'flops', 'latency'], 'Can only calculate overhead in area, flops or latency.'
        overhead = 0
        kept_ratio = torch.zeros([1, 22]).to(utils.net_device(net))
        for name, m in net.named_modules():
            if isinstance(m, BARStructuredWrapper):
                z = m.cal_mask(stochastic = False)
                alive_channels = (z > 0.).long().sum().item()
                if mode == 'latency':
                    kept_ratio[0][module_name.index(name)] = alive_channels / float(m.module.out_channels)
                partial_overhead = m.output_area * (z > 0.).long().sum().item()
                if mode == 'area':
                    overhead += partial_overhead
                elif mode == 'flops':
                    in_layers = get_previous_name(net, name)
                    alive_inchannels = get_alive_inchannels(net, in_layers, stochastic=False)
                    overhead += partial_overhead * alive_inchannels * (m.module.kernel_size[0] ** 2)
        if mode == 'latency':
            overhead = torch.clip(self.hardware_model(kept_ratio)[0], 0., 1.)
        return overhead

    def origin_overhead(self, net: nn.Module, mode:str='area') -> float:
        r"""
        Calculate the origin computation overhead before pruning.

        Args:
            net (nn.Module): The network to be calculated.
            mode(str): The format of overhead, could be area or flops.
        Returns:
            overhead (float): The origin computation overhead before pruning.
        """
        assert mode in ['area', 'flops', 'latency'], \
            'Can only calculate overhead in area, flops or latency.'
        if self._origin_overhead:
            return self._origin_overhead

        if mode == 'latency':
            self._origin_overhead = 1. # normalized latency.
            return self._origin_overhead

        self._origin_overhead = 0
        for name, m in net.named_modules():
            if isinstance(m, BARStructuredWrapper):
                m_ = m.module
                nchannels = m_.out_channels
                delta_overhead = m.output_area * nchannels
                if mode == 'flops':
                    delta_overhead *= (m_.kernel_size[0] ** 2) * m_.in_channels
                self._origin_overhead += delta_overhead
        return self._origin_overhead

    def sparsity_ratio(self, net: nn.Module, mode:str='area') -> float:
        r"""
        Calculate the spartial ratio.

        Args:
            net (nn.Module): The network to be calculated.
            mode(str): The format of overhead, could be area or flops.
        Returns:
            sparsity_ratio (float): The spartial ratio.
        """
        assert mode in ['area', 'flops', 'latency'], 'Can only calculate overhead in area, flops or latency.'
        current_overhead = self.current_overhead(net, mode)
        origin_overhead = self.origin_overhead(net, mode)
        sparsity_ratio = current_overhead / origin_overhead
        return sparsity_ratio

