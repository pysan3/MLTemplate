from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from torchvision import models

    from utils.manager import Manager


class ResNetBackBone(nn.Module):
    def __init__(self, resnet_type: str = "resnet101", return_layer=4, pretrained=True, veryverbose=False):
        super().__init__()
        self.veryverbose = veryverbose
        self.return_layer = return_layer
        self.main = self.load_resnet(resnet_type, pretrained)

    def load_resnet(self, resnet_type: str, pretrained=True) -> models.resnet.ResNet:
        from torchvision import models

        version = int(resnet_type[6:])
        if version not in {18, 34, 50, 101, 152}:
            raise RuntimeError(f"Unknown resnet_type: {version}")
        weights = None
        if pretrained:
            weights = getattr(models, f"ResNet{version}_Weights").DEFAULT
        model = getattr(models, f"resnet{version}")
        if self.veryverbose:
            print(f"Network weights: {weights}")
            print(f"ResNetBackBone: Loading {type(model)}")
        return model(weights=weights)

    def forward(self, x):
        x = self.main.conv1(x)
        x = self.main.bn1(x)
        x = self.main.relu(x)
        x = self.main.maxpool(x)
        x = self.main.layer1(x)
        if self.return_layer == 1:
            return x
        x = self.main.layer2(x)
        if self.return_layer == 2:
            return x
        x = self.main.layer3(x)
        if self.return_layer == 3:
            return x
        x = self.main.layer4(x)
        if self.return_layer == 4:
            return x
        raise NotImplementedError(f"return_layer must be <= 4. {self.return_layer=}")


class Network(nn.Module):
    def __init__(self, mgr: Manager, img_size=256):
        super().__init__()
        self.mgr = mgr
        self.num_out = self.mgr.MODULE.NUM_OUTPUT
        self.img_size = img_size
        self.bb = ResNetBackBone(self.mgr.MODULE.NAME, self.mgr.MODULE.USE_LAYER)
        channel = 2 ** (self.mgr.MODULE.USE_LAYER + 7)
        img_divide = 2 ** (self.mgr.MODULE.USE_LAYER + 1)
        self.bb_out_size = channel * (self.img_size // img_divide) * (self.img_size // img_divide)
        self.fc_out_size = self.num_out * 3
        self.main = nn.Sequential(
            nn.Linear(self.bb_out_size, self.fc_out_size),
        )
        self.report_network_summary = self.mgr.veryverbose

    def forward(self, img_batch: Tensor):
        if self.report_network_summary:
            from torch.profiler import ProfilerActivity, profile, record_function
            from torchinfo import summary

            print(f"Network.forward: batch.size={img_batch.shape[0]}, type={type(img_batch)}")
            print(f"img={img_batch.shape}, type={img_batch.dtype}")
            summary(self.bb, input_size=img_batch.shape, verbose=self.mgr.veryverbose)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
            ) as prof:
                with record_function("model_inference"):
                    result = self.bb(img_batch)
            print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
            if not self.mgr.FORCE_CPU:
                from torch.cuda import memory_summary

                print(memory_summary())
        else:
            result = self.bb(img_batch)
        result = result.view(-1, self.bb_out_size)
        result = self.main(result)
        result = result.view(-1, 3, self.num_out)
        if self.report_network_summary:
            print(f"{self.__class__.__name__}.forward: x={img_batch.shape} -> result={result.shape}")
        self.report_network_summary = False  # Show info only on the first time
        return result
