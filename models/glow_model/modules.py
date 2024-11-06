""" Modules for Glow Model """
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.glow_model.utils import compute_same_pad, split_feature

# from models.glow_model.model import FlowStep
# from data.datasets import postprocess
# MNIST dataset has 28x28 images, but the loss goes inf.
# def gaussian_p(mean, logs, x):
#     """
#     Compute the Gaussian log-probability.
#     lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
#     """
#     c = math.log(2 * math.pi)

#     # Ensure the spatial dimensions of all inputs match (B, C, H, W)
#     if x.size() != mean.size():
#         mean = F.interpolate(mean, size=(x.size(2), x.size(3)), mode='nearest')
#     if logs.size() != x.size():
#         logs = F.interpolate(logs, size=(x.size(2), x.size(3)), mode='nearest')

#     return -0.5 * (logs * 2.0 + ((x - mean) ** 2) / torch.exp(logs * 2.0) + c)

# def gaussian_likelihood(mean, logs, x):
#     """ Compute Gaussian likelihood and sum over the spatial and channel dimensions. """
#     p = gaussian_p(mean, logs, x)
    
#     # Sum over all but the batch dimension (dim=0)
#     return torch.sum(p, dim=[1, 2, 3])

# def gaussian_sample(mean, logs, temperature=1):
#     """ Sample from Gaussian with temperature scaling. """
#     # Ensure the spatial dimensions of mean and logs match
#     if logs.size() != mean.size():
#         logs = F.interpolate(logs, size=(mean.size(2), mean.size(3)), mode='nearest')

#     # Sample from Gaussian distribution with temperature
#     z = torch.normal(mean, torch.exp(logs) * temperature)

#     return z

def gaussian_p(mean, logs, x):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
            k = 1 (Independent)
            Var = logs ** 2
    """
    c = math.log(2 * math.pi)
    return -0.5 * (logs * 2.0 + ((x - mean) ** 2) / torch.exp(logs * 2.0) + c)


def gaussian_likelihood(mean, logs, x):
    """ Compute Gaussian likelihood """
    p = gaussian_p(mean, logs, x)
    return torch.sum(p, dim=[1, 2, 3])


def gaussian_sample(mean, logs, temperature=1):
    """ Sample from Gaussian with temperature """
    z = torch.normal(mean, torch.exp(logs) * temperature)
    return z

# def gaussian_sample(mean, logs, temperature=1.0):
#     # Obtain mean and logs from the model
#     # mean, logs = model(y_onehot=None, reverse=True)
    
#     # Clamp logs to avoid extreme values after exponentiation
#     logs = torch.clamp(logs, min=-10, max=10)
    
#     # Check for infinities and replace them
#     mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
#     logs = torch.where(torch.isfinite(logs), logs, torch.zeros_like(logs))

#     # # Print for debugging
#     # print("Mean values after replacement:", mean)
#     # print("Standard deviations after replacement:", torch.exp(logs) * temperature)

#     # Sample with adjusted mean and std
#     z = torch.normal(mean, torch.exp(logs) * temperature)
#     return z



# def check_for_nan(tensor, layer_name, layer_index):
#     if torch.isnan(tensor).any():
#         print(f"NaN detected in {layer_name} at index {layer_index}")

# def sample_and_debug_glow(model, z, temperature=1.0):
#     # Start reverse pass with latent `z` adjusted by temperature
#     z = z * temperature

#     for i, layer in enumerate(model.flow.layers):
#         if isinstance(layer, FlowStep):
#             # Run each component in the FlowStep separately to isolate `NaNs`

#             # ActNorm
#             z = layer.actnorm(z, reverse=True)
#             check_for_nan(z, "ActNorm", i)

#             # InvertibleConv1x1
#             z = layer.invconv(z, reverse=True)
#             check_for_nan(z, "InvertibleConv1x1", i)

#             # Convolutional Block
#             for j, sublayer in enumerate(layer.block):
#                 z = sublayer(z)
#                 check_for_nan(z, f"Conv Block Layer {j}", i)

#         elif isinstance(layer, Split2d):
#             # Split2d layer reverse pass
#             z = layer(z, reverse=True)
#             check_for_nan(z, "Split2d", i)

#         elif isinstance(layer, SqueezeLayer):
#             # SqueezeLayer reverse pass
#             z = layer(z, reverse=True)
#             check_for_nan(z, "SqueezeLayer", i)

#     # Final output
#     images = postprocess(z)
#     check_for_nan(images, "Final Output", -1)
#     return images


# def squeeze2d(inputs, factor):
#     """ Squeeze input 2D tensor into 4D """
#     if factor == 1:
#         return inputs

#     B, C, H, W = inputs.size()

#     # Pad H and W if not divisible by factor (for cases like 28x28 with factor 2)
#     if H % factor != 0 or W % factor != 0:
#         pad_h = factor - (H % factor) if H % factor != 0 else 0
#         pad_w = factor - (W % factor) if W % factor != 0 else 0
#         inputs = F.pad(inputs, (0, pad_w, 0, pad_h))

#         # Update H and W after padding
#         _, _, H, W = inputs.size()

#     assert H % factor == 0 and W % factor == 0, f"H={H}, W={W} not divisible by factor={factor}"

#     # Reshape and permute
#     x = inputs.view(B, C, H // factor, factor, W // factor, factor)
#     x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
#     x = x.view(B, C * factor * factor, H // factor, W // factor)

#     return x

# def unsqueeze2d(inputs, factor):
#     """ Unsqueeze 4D tensor into 2D """
#     if factor == 1:
#         return inputs

#     factor2 = factor ** 2

#     B, C, H, W = inputs.size()

#     assert C % factor2 == 0, f"Channels={C} not divisible by factor squared={factor2}"

#     # Reshape and permute
#     x = inputs.view(B, C // factor2, factor, factor, H, W)
#     x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
#     x = x.view(B, C // factor2, H * factor, W * factor)

#     # Remove any padding if added during squeeze
#     orig_H, orig_W = H * factor, W * factor
#     if orig_H != inputs.size(2) or orig_W != inputs.size(3):
#         x = x[:, :, :orig_H, :orig_W]

#     return x
def squeeze2d(inputs, factor):
    """ Squeeze input 2D tensor into 4D """
    if factor == 1:
        return inputs

    B, C, H, W = inputs.size()

    assert (
        H % factor == 0 and W % factor == 0
    ), f"H or W modulo factor is not 0: H={H}, W={W}, factor={factor}"

    x = inputs.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)

    return x


def unsqueeze2d(inputs, factor):
    """ Unsqueeze 4D tensor into 2D """
    if factor == 1:
        return inputs

    factor2 = factor**2

    B, C, H, W = inputs.size()

    assert C % (factor2) == 0, "C module factor squared is not 0"

    x = inputs.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)

    return x


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.0):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, inputs):
        """ Initialize the parameters """
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = -torch.mean(inputs.clone(), dim=[0, 2, 3], keepdim=True)
            var = torch.mean((inputs.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(var) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited = True

    def _center(self, inputs, reverse=False):
        if reverse:
            return inputs - self.bias
        return inputs + self.bias

    def _scale(self, inputs, logdet=None, reverse=False):
        if reverse:
            inputs = inputs * torch.exp(-self.logs)
        else:
            inputs = inputs * torch.exp(self.logs)

        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply by number of pixels
            """
            b, c, h, w = inputs.shape

            dlogdet = torch.sum(self.logs) * h * w

            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return inputs, logdet

    def forward(self, inputs, logdet=None, reverse=False):
        """ Forward and backward pass """
        self._check_input_dim(inputs)

        if not self.inited:
            self.initialize_parameters(inputs)

        if reverse:
            inputs, logdet = self._scale(inputs, logdet, reverse)
            inputs = self._center(inputs, reverse)
        else:
            inputs = self._center(inputs, reverse)
            inputs, logdet = self._scale(inputs, logdet, reverse)

        return inputs, logdet


class ActNorm2d(_ActNorm):
    """ Activation Normalization for 2D inputs """
    def __init__(self, num_features, scale=1.0):
        super().__init__(num_features, scale)

    def _check_input_dim(self, inputs):
        assert len(inputs.size()) == 4
        assert inputs.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, inputs.size()
            )
        )


class LinearZeros(nn.Module):
    """ Linear layer with zeros initialization """
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.logscale_factor = logscale_factor

        self.logs = nn.Parameter(torch.zeros(out_channels))

    def forward(self, inputs):
        """ Forward pass """
        output = self.linear(inputs)
        return output * torch.exp(self.logs * self.logscale_factor)


class Conv2d(nn.Module):
    """ Conv2d layer with ActNorm """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        do_actnorm=True,
        weight_std=0.05,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=(not do_actnorm),
        )

        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)

        self.do_actnorm = do_actnorm

    def forward(self, inputs):
        x = self.conv(inputs)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Module):
    """ Conv2d layer with zeros initialization """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        logscale_factor=3,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, inputs):
        """ Forward pass """
        output = self.conv(inputs)
        return output * torch.exp(self.logs * self.logscale_factor)


class Permute2d(nn.Module):
    """ Permute the input 2D tensor """
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        self.indices = torch.arange(self.num_channels - 1, -1, -1, dtype=torch.long)
        self.indices_inverse = torch.zeros((self.num_channels), dtype=torch.long)

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        """ Reset the indices """
        shuffle_idx = torch.randperm(self.indices.shape[0])
        self.indices = self.indices[shuffle_idx]

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, inputs, reverse=False):
        """ Forward and backward pass """
        assert len(inputs.size()) == 4

        if not reverse:
            inputs = inputs[:, self.indices, :, :]
            return inputs
        else:
            return inputs[:, self.indices_inverse, :, :]


class Split2d(nn.Module):
    """ Split the input 2D tensor """
    def __init__(self, num_channels):
        super().__init__()
        self.conv = Conv2dZeros(num_channels // 2, num_channels)

    def split2d_prior(self, z):
        """ Split the input 2D tensor """
        h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, inputs, logdet=0.0, reverse=False, temperature=None):
        """ Forward and backward pass """
        if reverse:
            z1 = inputs
            mean, logs = self.split2d_prior(z1)
            z2 = gaussian_sample(mean, logs, temperature)
            # z2 = sample_and_debug_glow(self, mean, temperature)
            z = torch.cat((z1, z2), dim=1)
            return z, logdet
        else:
            z1, z2 = split_feature(inputs, "split")
            mean, logs = self.split2d_prior(z1)
            logdet = gaussian_likelihood(mean, logs, z2) + logdet
            return z1, logdet


class SqueezeLayer(nn.Module):
    """ Squeeze the input 2D tensor """
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, inputs, logdet=None, reverse=False):
        """ Forward and backward pass """
        if reverse:
            output = unsqueeze2d(inputs, self.factor)
        else:
            output = squeeze2d(inputs, self.factor)

        return output, logdet


class InvertibleConv1x1(nn.Module):
    """ Invertible 1x1 Convolution """
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, inputs, reverse):
        """ Get the weight """
        b, c, h, w = inputs.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(self.lower.device)
            self.eye = self.eye.to(self.lower.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous().to(self.upper.device)
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1).to(inputs.device), dlogdet.to(inputs.device)

    def forward(self, inputs, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(inputs, reverse)

        if not reverse:
            z = F.conv2d(inputs, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(inputs, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet
