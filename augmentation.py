import torch
from torchvision import transforms


class RandomRollPct(torch.nn.Module):
    def __init__(self, percent, axis=2, pct_apply=0.8):
        super().__init__()
        self.axis = axis
        self.pct_apply = pct_apply

        assert len(percent) == 2
        self.percent = percent
        self.mode = "percent"

    def forward(self, image):
        if torch.rand(1) < self.pct_apply:
            translate_x = self.percent[0] + (
                torch.rand(1) * (self.percent[0] - self.percent[1])
            )
            translate_x_px = translate_x * image.shape[-1]

            return torch.roll(image, (int(translate_x_px)), self.axis)
        else:
            return image


class RandomRollPx(torch.nn.Module):
    def __init__(self, percent=None, px=None, shape=None, axis=2, pct_apply=0.8):
        super().__init__()
        self.axis = axis
        assert len(px) == 2
        self.px = px
        self.pct_apply = pct_apply

    def forward(self, image):
        if torch.rand(1) < self.pct_apply:
            translate_x = self.px[0] + (torch.rand(1) * (self.px[0] - self.px[1]))
            translate_x_px = translate_x.int32()

            return torch.roll(image, translate_x_px, self.axis)
        else:
            return image


class RandomMultiplyChannelwise(torch.nn.Module):
    def __init__(self, mul=(0.8, 1.2), pct_apply=0.8):
        super().__init__()
        assert len(mul) == 2
        self.mul = mul
        self.pct_apply = pct_apply

    def forward(self, image):
        if torch.rand(1) < self.pct_apply:
            mul_tensor = self.mul[0] + (
                torch.rand(image.shape[-3]) * (self.mul[1] - self.mul[0])
            )
            out_tensor = torch.einsum("...ijk, ...i -> ...ijk", image, mul_tensor)
            # mul = [random.uniform(self.mul[0], self.mul[1]) for _ in range(image.shape[0])]
            # mult_tensor = torch.stack([x * image[i,:,:] for i,x in enumerate(mul)])
            return out_tensor
        else:
            return image


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, pct=0.5):
        super().__init__()
        self.pct = pct

    def forward(self, image):
        if torch.rand(1) < self.pct:
            return torch.fliplr(image)

        else:
            return image


def get_default_augmenter():
    my_transforms = transforms.RandomApply(
        torch.nn.ModuleList(
            [
                RandomHorizontalFlip(0.8),
                RandomRollPct(percent=(-0.5, 0.5), axis=2, pct_apply=0.8),
                RandomMultiplyChannelwise((0.8, 1.2), pct_apply=0.8),
            ]
        ),
        0.7,
    )

    scripted_transforms = torch.jit.script(my_transforms)
    return scripted_transforms


def test_augs_work():
    plottable_mat = torch.cumsum(torch.diag(torch.ones((10))), 1)
    img_3c = torch.stack([plottable_mat for _ in range(3)])

    # Note that ir runs only on py3.8
    my_transforms = transforms.RandomApply(
        torch.nn.ModuleList(
            [
                RandomHorizontalFlip(0.5),
                RandomRollPct(percent=(-0.5, 0.5), axis=2),
                RandomMultiplyChannelwise((0.1, 20)),
            ]
        ),
        0.3,
    )

    scripted_transforms = torch.jit.script(my_transforms)
    return scripted_transforms(img_3c)


if __name__ == "__main__":
    print(test_augs_work())
