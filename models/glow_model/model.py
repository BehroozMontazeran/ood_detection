""" Glow Model for PyTorch """

import json
import math
import re
from os import listdir, path

import torch
import torch.nn as nn

from models.glow_model.modules import (ActNorm2d, Conv2d, Conv2dZeros,
                                       InvertibleConv1x1, LinearZeros,
                                       Permute2d, Split2d, SqueezeLayer,
                                       gaussian_likelihood, gaussian_sample)
from models.glow_model.utils import split_feature, uniform_binning_correction
# from ood_scores.ood_extractors import GenerativeModel
from utilities.routes import GLOW_ROOT
from utilities.utils import get_image_shape

N_BITS = 8

checkpoint_regex = re.compile("glow_checkpoint_\d*.pt")

def postprocess(x):
    """Deprecated."""
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2**N_BITS
    return torch.clamp(x, 0, 255).byte()


def get_block(in_channels, out_channels, hidden_channels):
    block = nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
        nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels),
    )
    return block

class FlowStep(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
    ):
        super().__init__()
        self.flow_coupling = flow_coupling

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (
                self.shuffle(z, rev),
                logdet,
            )
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (
                self.reverse(z, rev),
                logdet,
            )

        # 3. coupling
        if flow_coupling == "additive":
            self.block = get_block(in_channels // 2, in_channels // 2, hidden_channels)
        elif flow_coupling == "affine":
            self.block = get_block(in_channels // 2, in_channels, hidden_channels)

    def forward(self, inputs, logdet=None, reverse=False):
        """ forward and backward pass """
        if not reverse:
            return self.normal_flow(inputs, logdet)
        else:
            return self.reverse_flow(inputs, logdet)

    def normal_flow(self, inputs, logdet):
        """ forward pass """
        assert inputs.size(1) % 2 == 0

        # 1. actnorm
        z, logdet = self.actnorm(inputs, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, inputs, logdet):
        """ reverse pass """
        assert inputs.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = split_feature(inputs, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    """ Flow Network """
    def __init__(
        self,
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K
        self.L = L

        H, W, C = image_shape

        for i in range(L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])

            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                    )
                )
                self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if i < L - 1:
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, inputs, logdet=0.0, reverse=False, temperature=None):
        """ forward and backward pass """
        if reverse:
            return self.decode(inputs, temperature)
        else:
            return self.encode(inputs, logdet)

    def encode(self, z, logdet=0.0):
        """ forward pass """
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        """ reverse pass """
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z


class Glow(nn.Module):#, GenerativeModel):
    """ Glow Model """
    def __init__(
        self,
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        y_classes,
        learn_top,
        y_condition,
    ):
        super().__init__()
        self.flow = FlowNet(
            image_shape=image_shape,
            hidden_channels=hidden_channels,
            K=K,
            L=L,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
        )
        self.y_classes = y_classes
        self.y_condition = y_condition

        self.learn_top = learn_top

        # learned prior
        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        if y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(y_classes, 2 * C)
            self.project_class = LinearZeros(C, y_classes)

        self.register_buffer(
            "prior_h",
            torch.zeros(
                [
                    1,
                    self.flow.output_shapes[-1][1] * 2,
                    self.flow.output_shapes[-1][2],
                    self.flow.output_shapes[-1][3],
                ]
            ),
        )

    def prior(self, data, y_onehot=None):
        """ get prior of z """
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            # Hardcoded a batch size of 32 here
            h = self.prior_h.repeat(32, 1, 1, 1)

        channels = h.size(1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            h += yp.view(h.shape[0], channels, 1, 1)

        return split_feature(h, "split")

    def forward(self, x=None, y_onehot=None, z=None, temperature=None, reverse=False):
        """ forward and backward pass """
        if reverse:
            return self.reverse_flow(z, y_onehot, temperature)
        else:
            return self.normal_flow(x, y_onehot)

    def normal_flow(self, x, y_onehot):
        """ forward pass """
        b, c, h, w = x.shape

        x, logdet = uniform_binning_correction(x)

        z, objective = self.flow(x, logdet=logdet, reverse=False)

        mean, logs = self.prior(x, y_onehot)
        objective += gaussian_likelihood(mean, logs, z)

        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # Full objective - converted to bits per dimension
        bpd = (-objective) / (math.log(2.0) * c * h * w)

        return z, bpd, y_logits

    def reverse_flow(self, z, y_onehot, temperature):
        """ reverse pass """
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z, y_onehot)
                z = gaussian_sample(mean, logs, temperature)
            x = self.flow(z, temperature=temperature, reverse=True)
        return x

    def set_actnorm_init(self):
        """ set actnorm layers inited """
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True

    def eval_nll(self, x):
        """ evaluate NLL (bits/dim) """
        _, nll, _ = self.forward(x)
        return nll

    def generate_sample(self, batch_size):
        """ generate samples """
        # Batch size is fixed at 32
        if batch_size != 32:
            raise NotImplementedError("currently glow only supports batch sizes of 32.")
        imgs = self.forward(temperature=1, reverse=True)
        imgs = imgs.clamp(-0.5, 0.5)
        # return postprocess(imgs)
        return imgs

    @staticmethod
    def load_serialised(model_name):
        # if "num_classes" not in params:
        #     num_classes = 10
        # elif "image_shape" not in params:
        #     image_shape = (32, 32, 3)

        device = "cuda"

        base_path = path.join(GLOW_ROOT, model_name)

        print(f"loading model from: {base_path}")

        params_path = path.join(base_path, "hparams.json")

        with open(params_path) as json_file:
            hparams = json.load(json_file)

        image_shape = get_image_shape(hparams["dataset"])
        num_classes = None

        model = Glow(
            image_shape,
            hparams["hidden_channels"],
            hparams["K"],
            hparams["L"],
            hparams["actnorm_scale"],
            hparams["flow_permutation"],
            hparams["flow_coupling"],
            hparams["LU_decomposed"],
            num_classes,
            hparams["learn_top"],
            hparams["y_condition"],
        )

        torch_filename = None

        for filename in listdir(base_path):

            m = checkpoint_regex.match(filename)
            if m:
                torch_filename = m.group()

        model_path = path.join(base_path, torch_filename)

        print(f"model_path: {model_path}")

        state_dicts = torch.load(model_path, map_location=device)
        print(f"stored information: {state_dicts.keys()}")

        model.load_state_dict(
            state_dicts["model"]
        )  # You need to direct it "model" part of the file

        model.set_actnorm_init()

        model = model.to(device)

        model = model.eval()

        return model

    @staticmethod
    def get_save_file(name):
        raise NotImplementedError()

    @staticmethod
    def get_save_dir(name):
        raise NotImplementedError()

    @staticmethod
    def get_params(name):
        raise NotImplementedError()

