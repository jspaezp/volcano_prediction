from numpy.lib.twodim_base import mask_indices
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.models as models


def get_total_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    return pytorch_total_params, pytorch_trainable_params


class densenet_10r(models.densenet.DenseNet):
    """
    densenet_10r Modified version of DenseNet

    It modifies DenseNet to take 10 input channels and do regression instead of
    classification
    """

    __doc__ += models.densenet.DenseNet.__doc__

    def __init__(self, num_init_features: int = 64, *args, **kwargs) -> None:
        super(densenet_10r, self).__init__(
            num_init_features=num_init_features, num_classes=1, *args, **kwargs
        )
        self.features.conv0 = nn.Conv2d(
            10,
            num_init_features,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.rel = nn.LeakyReLU()

    def forward(self, x):
        x = super(densenet_10r, self).forward(x)
        x = self.rel(x)
        return x


def test_densenet_10r():
    # Tests that a 10c input works on the network
    x_image = torch.randn(1, 10, 224, 224)
    for _ in range(5):
        model = densenet_10r()
        ouput = model(x_image)
    densenet169 = densenet_10r(
        **{"growth_rate": 32, "block_config": (6, 12, 32, 32), "num_init_features": 64}
    )
    print(">>>>> densenet169")
    print(densenet169)
    print(get_total_parameters(densenet169))


class resnet_10c(models.resnet.ResNet):
    def __init__(
        self,
        block=models.resnet.BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=4,
        *args,
        **kwargs
    ):
        self.inplanes = 64
        super(resnet_10c, self).__init__(
            block, layers, num_classes=num_classes, *args, **kwargs
        )

        # This makes the input be 10 channels and not 3
        self.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)


class resnet_10r(resnet_10c):
    def __init__(
        self, block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], *args, **kwargs
    ):
        self.inplanes = 64
        super(resnet_10r, self).__init__(block, layers, num_classes=1, *args, **kwargs)
        self.lin = nn.Linear(self.fc.out_features, 1)
        self.rel = nn.LeakyReLU()

    def forward(self, x):
        x = super(resnet_10r, self).forward(x)
        x = self.lin(x)
        x = self.rel(x)
        return x


def test_resnet_10r():
    # Tests that a 10c input works on the network
    x_image = torch.randn(1, 10, 224, 224)
    for _ in range(5):
        model = resnet_10r()
        output = model(x_image)

    model = resnet_10r()
    output = model(x_image)
    assert len(output) == 1
    resnet50 = resnet_10r(**{"layers": [3, 4, 6, 3]})
    print(">>>>> resnet50")
    print(resnet50)
    print(get_total_parameters(resnet50))


def test_resnet_10c():
    resnet_10c(layers=[2, 3, 4, 5])
    x_image = torch.randn(1, 10, 224, 224)
    model = resnet_10c()
    ouput = model(x_image)


if __name__ == "__main__":
    test_resnet_10c()
    test_resnet_10r()
    test_densenet_10r()
