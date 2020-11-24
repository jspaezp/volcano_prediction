import torch
import matplotlib.pyplot as plt


def show_tensor(a):
    if len(a.shape) != 3:
        print(f"Provided tensor is not 3d {a.shape}, will try to make it such ...")
        a = a[0,:,:,:]

    if a.shape[0] > 100:
        print("Probably you did not want to pass this array ...")
    # Tensor of shape (channels, w, h)
    linear_tensor = torch.cat([a[x, :, :] for x in range(a.shape[0])],1) 
    plt.imshow(-linear_tensor)
    plt.show()

def test_show_tensor():
    a = torch.rand((10, 50, 50))
    show_tensor(a)
    a = torch.rand((1, 10, 50, 50))
    show_tensor(a)

if __name__ == "__main__":
    test_show_tensor()