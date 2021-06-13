import torch.onnx
from Net import Net


IMG_SIZE = 60

if __name__ == "__main__":
    net = Net()
    net.load_state_dict(torch.load("Net_100.pth"))
    net.eval()

    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(net, dummy_input, "Net_100.onnx")

