from numpy.lib.twodim_base import mask_indices
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.models as models


class resnet_10c(models.resnet.ResNet):
    def __init__(
        self, block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=4
    ):
        self.inplanes = 64
        super(resnet_10c, self).__init__(block, layers, num_classes=num_classes)

        # This makes the input be 10 channels and not 3
        self.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)


class resnet_10r(resnet_10c):
    def __init__(self, block=models.resnet.BasicBlock, layers=[2, 2, 2, 2]):
        self.inplanes = 64
        super(resnet_10r, self).__init__(block, layers, num_classes=1)
        self.lin = nn.Linear(self.fc.out_features, 1)
        self.rel = nn.ReLU()

    def forward(self, x):
        x = super(resnet_10r, self).forward(x)
        x = self.lin(x)
        x = self.rel(x)
        return x


#
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        image_modules = list(resnet_10c().children())[
            :-1
        ]  # all layer expect last layer
        self.modelA = nn.Sequential(*image_modules)

    def forward(self, image):
        a = self.modelA(image)
        x = nnf.log_softmax(a)
        return x


if __name__ == "__main__":

    x_image = torch.randn(1, 10, 224, 224)

    model = MyModel()
    model.fc = nn.Linear(2048, 1)

    print(">>>>>>> Out for MyModel, replacing last layer")
    ouput = model(x_image)
    print(ouput.shape)

    print(">>>>>>> Out for resnet 10c")
    model = resnet_10c()
    ouput = model(x_image)
    print(ouput)

    print(">>>>>>> Out for resnet 10r")
    for _ in range(20):
        model = resnet_10r()
        ouput = model(x_image)
        print(ouput)
    # print(model.parameters)
