import os
import torch

from torchvision.models import (
    VGG16_BN_Weights,
    VGG16_Weights,
    VGG19_BN_Weights,
    VGG19_Weights,
    vgg16,
    vgg19,
)
from torchvision.models.vgg import vgg16_bn, vgg19_bn


def save(model, path):
    file = os.path.expanduser(path)

    dir = os.path.dirname(file)

    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Directory `{dir}` created.")

    model_weights = model.state_dict()
    torch.save(model_weights, file)

    print(f"saved to `{file}`.")


save(
    vgg16(weights=VGG16_Weights.DEFAULT),
    "~/.cache/vgg-burn/vgg16-397923af.pth",
)

save(
    vgg16_bn(weights=VGG16_BN_Weights.DEFAULT),
    "~/.cache/vgg-burn/vgg16_bn-6c64b313.pth",
)

save(
    vgg19(weights=VGG19_Weights.DEFAULT),
    "~/.cache/vgg-burn/vgg19-dcbb9e9d.pth",
)

save(
    vgg19_bn(weights=VGG19_BN_Weights.DEFAULT),
    "~/.cache/vgg-burn/vgg19_bn-c79401a0.pth",
)
