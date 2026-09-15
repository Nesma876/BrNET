"""Synthetic one-step smoke only; not a scientific experiment or performance result."""
import argparse
import json
from pathlib import Path
import torch
from src.models.brnet import BrNet
from src.training.seed import seed_everything

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    seed_everything(42)
    model=BrNet(3)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)
    loss=torch.nn.functional.cross_entropy(model.logits(torch.rand(2,1,256,256)),torch.tensor([0,1]))
    assert torch.isfinite(loss)
    loss.backward();optimizer.step()
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f: json.dump({'synthetic_only':True,'finite_loss':True,'seed':42,'classes':3},f)

if __name__=='__main__':main()
