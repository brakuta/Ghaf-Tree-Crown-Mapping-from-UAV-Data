

from transformers import AutoModel
from PIL import Image
from timm.data.transforms_factory import create_transform
import torch.nn as nn
import timm
from mmseg.registry import MODELS
import torch
from torch import nn

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mambavision import create_model
from mmengine.logging.history_buffer import HistoryBuffer
from numpy.core.multiarray import _reconstruct
import torch

# Monkey-patch torch.load within MMEngine’s loader
torch.serialization.add_safe_globals([HistoryBuffer])
_orig_torch_load = torch.load
# 2. Define the patched loader
def _patched_torch_load(filename, map_location=None, *args, **kwargs):
    # Force weights_only=False exactly once
    kwargs.pop('weights_only', None)
    return _orig_torch_load(
        filename,
        map_location=map_location,
        weights_only=False,
        *args,
        **kwargs
    )

torch.load = _patched_torch_load
# torch.serialization.add_safe_globals([_reconstruct])

class mamba_vision_tim(nn.Module):
    def __init__(self):
        super(mamba_vision_tim, self).__init__()  # Ensure nn.Module is properly initialized
        # self.backbone = timm.create_model(
        #     'coatnet_3_rw_224.sw_in12k',
        #     pretrained=True,
        #     features_only=True,
        #     img_size=1024
        # )
        # self.backbone = timm.create_model(
        # 'mambaout_base.in1k',
        # pretrained=True,
        # features_only=True,
        # )
        # self.backbone = create_model('mamba_vision_T', pretrained=False)
        # self.backbone.train()
        #, model_path="/tmp/mambavision_tiny_1k.pth.tar")#, model_path="/tmp/mambavision_base_1k.pth.tar")
        self.backbone = AutoModel.from_pretrained("nvidia/MambaVision-B-1K", trust_remote_code=True)
        # self.backbone.train()

        # return model
        # self.backbone = AutoModel.from_pretrained("nvidia/MambaVision-L3-512-21K", trust_remote_code=True)
        # self.backbone = timm.create_model(
        #             'dpn98.mx_in1k',
        #             pretrained=True,
        #             features_only=True,


        #         )


    def forward(self, x):
        features = self.backbone(x)
        # print(features)

        # print("Size of extracted features in stage 0:", features[1][0].size())

        # print("Size of extracted features in stage 1:", features[1][1].size())

        # print("Size of extracted features in stage 2:", features[1][2].size())
        # print("Size of extracted features in stage 3:", features[1][3].size())

        return features[1]

@MODELS.register_module()
class mamba_small_vision_timm(mamba_vision_tim):
    def __init__(self, **kwargs):
        super(mamba_small_vision_timm, self).__init__()

        

