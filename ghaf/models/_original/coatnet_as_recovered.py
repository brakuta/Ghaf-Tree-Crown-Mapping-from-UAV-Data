import timm
from mmseg.registry import MODELS
import torch
from torch import nn
class coatnet_tim(nn.Module):
    def __init__(self):
        super(coatnet_tim, self).__init__()  # Ensure nn.Module is properly initialized
        # self.backbone = timm.create_model(
        #     'coatnet_3_rw_224.sw_in12k',
        #     pretrained=True,
        #     features_only=True,
        #     img_size=1024
        # )
        # self.backbone = timm.create_model(
        #     'dpn98.mx_in1k',
        #     pretrained=True,
        #     features_only=True,
        # )
        self.backbone = timm.create_model(
        'nextvit_base.bd_in1k',
        pretrained=True,
        features_only=True,
        )

    def forward(self, x):
        return self.backbone(x)
@MODELS.register_module()
class coatnet_small_timm(coatnet_tim):
    def __init__(self, **kwargs):
        super(coatnet_small_timm, self).__init__()
