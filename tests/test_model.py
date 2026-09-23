import torch
from src.models.brnet import BrNet
from src.training.seed import seed_everything

def test_reference_shapes_and_parameters():
    model = BrNet()
    assert sum(p.numel() for p in model.parameters()) == 895812
    x = torch.zeros(1, 1, 256, 256)
    spatial = []
    for layer in model.features:
        x = layer(x)
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
            spatial.append(x.shape[-1])
    assert spatial == [254, 127, 125, 62, 60, 30, 28, 14]
    assert model.features.conv4.out_channels == 64
    assert torch.allclose(model(torch.zeros(1,1,256,256)).sum(1), torch.ones(1))

def test_three_class_adaptation():
    assert sum(p.numel() for p in BrNet(3).parameters()) == 895747

def test_seed_determinism():
    seed_everything(42); a = BrNet().dense.weight.detach().clone()
    seed_everything(42); b = BrNet().dense.weight.detach().clone()
    assert torch.equal(a, b)
