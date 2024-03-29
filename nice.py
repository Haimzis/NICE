"""NICE model
"""
import torch
import torch.nn as nn
from torch.distributions.transforms import Transform, SigmoidTransform, AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np


"""Additive coupling layer.
"""
class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an additive coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()
        self.mask_config = mask_config
        self.in_out_dim = in_out_dim
        input_dim=in_out_dim // 2 + (in_out_dim % 2 > 0) * self.mask_config
        output_dim=in_out_dim // 2 + (in_out_dim % 2 > 0) * (1 - self.mask_config)
        layers = [nn.Linear(input_dim, mid_dim), nn.ReLU()]
        layers.extend(nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU()) for _ in range(hidden - 1))
        layers.append(nn.Linear(mid_dim, output_dim))
        self.transform = nn.Sequential(*layers)

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        x1, x2 = x[:, self.mask_config::2], x[:, 1 - self.mask_config::2]
        if reverse:
            x2 = x2 - self.transform(x1)
        else:
            x2 = x2 + self.transform(x1)
        ordered_concat = [x1, x2] if self.mask_config == 0 else [x2, x1]
        y = torch.stack(ordered_concat, dim=2).view(-1, self.in_out_dim)
        return y, log_det_J
    

class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.
        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()
        self.mask_config = mask_config
        self.in_out_dim = in_out_dim
        input_dim=in_out_dim // 2 + (in_out_dim % 2 > 0) * self.mask_config
        output_dim=in_out_dim // 2 + (in_out_dim % 2 > 0) * (1 - self.mask_config)

        layers = [nn.Linear(input_dim, mid_dim), nn.ReLU()]
        layers.extend(nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU()) for _ in range(hidden - 1))
        layers.extend([nn.Linear(mid_dim, output_dim), nn.Tanh()])
        self.scale_net = nn.Sequential(*layers)
        
        layers = [nn.Linear(input_dim, mid_dim), nn.ReLU()]
        layers.extend(nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU()) for _ in range(hidden - 1))
        layers.extend([nn.Linear(mid_dim, output_dim), nn.Tanh()])
        self.shift_net = nn.Sequential(*layers)

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        x1, x2 = x[:, self.mask_config::2], x[:, 1 - self.mask_config::2]

        s = torch.exp(self.scale_net(x1))
        t = self.shift_net(x1)

        if reverse:
            x2 = (x2 - t) / s**-1
        else:
            x2 = s * x2 + t
            log_det_J += s.log().sum(dim=1)

        ordered_concat = [x1, x2] if self.mask_config == 0 else [x2, x1]
        y = torch.stack(ordered_concat, dim=2).view(-1, self.in_out_dim)

        return y, log_det_J
    

"""Log-scaling layer.
"""
class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        scale = torch.exp(self.scale) + self.eps
        if reverse:
            x = x / scale
        else:
            x = x * scale
        log_det_J = torch.sum(torch.log(scale), dim=1)            
        return x, log_det_J


"""Standard logistic distribution.
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logistic = TransformedDistribution(Uniform(torch.tensor(0.).to(device), torch.tensor(1.).to(device)), [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])


"""NICE main model.
"""
class NICE(nn.Module):
    def __init__(self, prior, coupling, coupling_type,
                 in_out_dim, mid_dim, hidden, device):
        """Initialize a NICE.

        Args:
            coupling_type: 'additive' or 'adaptive'
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        self.device = device
        if prior == 'gaussian':
            self.prior = torch.distributions.Normal(
                torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        elif prior == 'logistic':
            self.prior = logistic
        else:
            raise ValueError('Prior not implemented.')
        self.in_out_dim = in_out_dim
        self.coupling = coupling
        self.coupling_type = coupling_type

        self.layers = nn.ModuleList()
        for i in range(coupling):
            if coupling_type == 'additive':
                self.layers.append(AdditiveCoupling(in_out_dim, mid_dim, hidden, i % 2))
            elif coupling_type == 'affine':
                self.layers.append(AffineCoupling(in_out_dim, mid_dim, hidden, i % 2))
            else:
                raise ValueError('Coupling type not implemented.')
        self.scaling_layer = Scaling(in_out_dim)

    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, _ = self.scaling_layer(z, reverse=True)
        for layer in reversed(list(self.layers)):
            x, _ = layer(x, 0, reverse=True) # jacobian is not needed for sampling
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        log_det_J = 0
        for layer in self.layers:
            x, log_det_J = layer(x, log_det_J)
        z, log_det_J_scale = self.scaling_layer(x)
        return z, log_det_J + log_det_J_scale

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_det_J -= np.log(256) * self.in_out_dim  # log det for rescaling from [0.256] (after dequantization) to [0,1]
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)
        x = self.f_inverse(z)
        return x

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)