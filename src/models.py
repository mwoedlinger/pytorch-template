import torch
import torch.nn as nn
from torch.autograd import Function
from .utils import Dict

__all__ = ['DebugModel', 'Baseline']

def calc_rate(y_q, mean, scale, sigma_lower_bound=0.1, likelihood_lower_bound=1e-9, offset=0.5, per_channel=False):
	"""
	Rate loss estimation of quantised latent variables using the provided CDF function (default = Laplacian CDF)
	Computation is performed per batch (across, channels, height, width), i.e. return shape is [BATCH]
	"""
	scale = LowerBound.apply(scale, sigma_lower_bound)
	y_q0 = y_q - mean
	y_q0 = y_q0.abs()
	upper = laplace_cdf(( offset - y_q0) / scale)
	lower = laplace_cdf((-offset - y_q0) / scale)
	likelihood = upper - lower
	likelihood = LowerBound.apply(likelihood, likelihood_lower_bound)

	if per_channel:
		total_bits = -torch.sum(torch.log2(likelihood), dim=(-1, -2))
	else:
		total_bits = -torch.sum(torch.log2(likelihood), dim=(-1, -2, -3))
	return total_bits

def quantise(x, mean, training):
    if training:
        return x + torch.zeros_like(x, device=x.device).uniform_(-0.5, 0.5)
    else:
        return torch.round(x - mean) + mean
    
class LaplaceCDF(torch.autograd.Function):
	"""
	CDF of the Laplacian distribution.
	"""

	@staticmethod
	def forward(ctx, x):
		s = torch.sign(x)
		expm1 = torch.expm1(-x.abs())
		ctx.save_for_backward(expm1)
		return 0.5 - 0.5 * s * expm1

	@staticmethod
	def backward(ctx, grad_output):
		expm1, = ctx.saved_tensors
		return 0.5 * grad_output * (expm1 + 1)

def _standard_cumulative_laplace(input):
	"""
	CDF of the Laplacian distribution.
	"""
	return LaplaceCDF.apply(input)

def laplace_cdf(input):
	""" 
	Computes CDF of standard Laplace distribution
	"""
	return _standard_cumulative_laplace(input)

class LowerBound(Function):
	""" Applies a lower bounded threshold function on to the inputs
		ensuring all scalars in the input >= bound.
		
		Gradients are propagated for values below the bound (as opposed to
		the built in PyTorch operations such as threshold and clamp)
	"""

	@staticmethod
	def forward(ctx, inputs, bound):
		b = torch.ones(inputs.size(), device=inputs.device, dtype=inputs.dtype) * bound
		ctx.save_for_backward(inputs, b)
		return torch.max(inputs, b)

	@staticmethod
	def backward(ctx, grad_output):
		inputs, b = ctx.saved_tensors

		pass_through_1 = inputs >= b
		pass_through_2 = grad_output < 0

		pass_through = pass_through_1 | pass_through_2
		return pass_through.type(grad_output.dtype) * grad_output, None

class Baseline(nn.Module):

    def __init__(self, in_channels=3, N=192, M=12):
        super().__init__()
        
        self.E = nn.Sequential(
            nn.Conv2d(in_channels, N, 3, 2, 1),					                # 1
            nn.PReLU(N, init=0.2),
            nn.Conv2d(N, N, 3, 2, 1),							                # 2
            nn.PReLU(N, init=0.2),
            nn.Conv2d(N, N, 3, 2, 1),							                # 4
            nn.PReLU(N, init=0.2),
            nn.Conv2d(N, M, 3, 1, 1)							                # 8
        )

        self.D = nn.Sequential(
            nn.ConvTranspose2d(M, N, 3, 2, 1, 1),				                # 8
            nn.PReLU(N, init=0.2),
            nn.ConvTranspose2d(N, N, 3, 2, 1, 1),				                # 4
            nn.PReLU(N, init=0.2),
            nn.ConvTranspose2d(N, N, 3, 2, 1, 1),				                # 2
            nn.PReLU(N, init=0.2),
            nn.ConvTranspose2d(N, in_channels, 3, 1, 1, 0)                      # 1
        )
    
        self.HE = nn.Sequential(
            nn.Conv2d(M, N, 3, 2, 1),							                # 8
            nn.PReLU(N, init=0.2),
            nn.Conv2d(N, N, 3, 2, 1),							                # 16
            nn.PReLU(N, init=0.2),
            nn.Conv2d(N, M, 3, 1, 1)							                # 32
        )

        self.HD = nn.Sequential(
            nn.ConvTranspose2d(M, N, 3, 2, 1, 1),				                # 32
            nn.PReLU(N, init=0.2),
            nn.ConvTranspose2d(N, N, 3, 2, 1, 1),				                # 16
            nn.PReLU(N, init=0.2),
            nn.Conv2d(N, 2*M, 3, 1, 1),				                            # 8
        )

        self.z_loc = nn.Parameter(torch.empty((1, M, 1, 1)))
        self.z_loc.data.normal_(0.0, 1.0)

        self.z_scale = nn.Parameter(torch.empty((1, M, 1, 1)))
        self.z_scale.data.uniform_(1.0, 1.5)

    def entropy(self, y):
        z = self.HE(y)     

        # Quantise z
        z_hat = quantise(z, self.z_loc, training=self.training)

        # Compute probability parameters for y
        y_entropy = self.HD(z_hat)
        y_loc, y_scale = torch.chunk(y_entropy, chunks=2, dim=1)

        # Quantise y
        y_hat = quantise(y, y_loc, training=self.training)

        return Dict(
                    y_hat=y_hat,
                    y_loc=y_loc,
                    y_scale=y_scale,

                    z_hat=z_hat,
                    z_loc=self.z_loc,
                    z_scale=self.z_scale
                    )

    def forward(self, x):
        # forward pass through model
        y = self.E(x)
        latents = self.entropy(y)
        x_hat = self.D(latents.y_hat)

        # Calculate rates for z and y
        bpp_z = calc_rate(latents.z_hat, latents.z_loc, latents.z_scale)
        bpp_y = calc_rate(latents.y_hat, latents.y_loc, latents.y_scale)

        return Dict(
            latents=latents,
            pred=x_hat,
            rate=Dict(
                y=bpp_y,
                z=bpp_z
            )
        )

class DebugModel(Baseline):

    def __init__(self, in_channels=3, M=6):
        super().__init__(in_channels=in_channels, M=M)
        s = 'DEBUG MODEL IS USED!'
        print('#'*(len(s) + 4) + f'\n# {s} #\n' + '#'*(len(s) + 4))

        self.E = nn.Conv2d(in_channels, M, 3, 1, 1)
        self.D = nn.Conv2d(M, in_channels, 3, 1, 1)
        self.HE = nn.Conv2d(M, M, 3, 1, 1)
        self.HD = nn.Conv2d(M, 2*M, 3, 1, 1)