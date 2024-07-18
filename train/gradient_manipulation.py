from torch.nn import Module

from train.gradient_manipulation_functionals import gsl, grl


class GradientReversalLayer(Module):
    def __init__(self, grad_scaling):
        """
        Gradient reversal layer
        Unsupervised Domain Adaptation by Backpropagation - Yaroslav Ganin, Victor Lempitsky
        https://arxiv.org/abs/1409.7495

        :param grad_scaling: the scaling factor that should be applied on the gradient in the backpropagation phase
        """
        super().__init__()
        self.grad_scaling = grad_scaling

    def forward(self, input):
        return grl(input, self.grad_scaling)

    def extra_repr(self) -> str:
        return f"grad_scaling={self.grad_scaling}"


class GradientScalingLayer(Module):
    def __init__(self, grad_scaling_values):
        super().__init__()
        self.grad_scaling_values = grad_scaling_values

    def forward(self, input, idxs):
        scaling = self.grad_scaling_values[idxs]
        return gsl(input, scaling)

    def extra_repr(self) -> str:
        return "grad_scaling"

    def to(self, *args, **kwargs):
        self.grad_scaling_values = self.grad_scaling_values.to(*args, **kwargs)
        return super().to(*args, **kwargs)
