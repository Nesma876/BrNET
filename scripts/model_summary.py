import csv
import json
from pathlib import Path
import torch
from src.models.brnet import BrNet

def run():
    model = BrNet().eval()
    rows, handles = [], []
    def hook(name):
        def record(layer, args, output):
            rows.append(dict(layer=name, type=type(layer).__name__, input_shape=list(args[0].shape),
                output_shape=list(output.shape), kernel=str(getattr(layer, 'kernel_size', '')),
                stride=str(getattr(layer, 'stride', '')), padding=str(getattr(layer, 'padding', '')),
                trainable_parameters=sum(p.numel() for p in layer.parameters() if p.requires_grad)))
        return record
    for name, layer in model.named_modules():
        if name and not list(layer.children()):
            handles.append(layer.register_forward_hook(hook(name)))
    with torch.no_grad():
        model(torch.zeros(1, 1, 256, 256))
    for handle in handles:
        handle.remove()
    total = sum(p.numel() for p in model.parameters())
    with open('results/brnet_model_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    Path('reports/BRNET_MODEL_SUMMARY.md').write_text('# BrNet actual forward-pass audit\n\n'
        + f'Reference four-class parameters: **{total:,}**. Expected: 895,812.\n\n'
        + '```json\n' + json.dumps(rows, indent=2) + '\n```\n\n'
        + 'Grad-CAM candidate: features.conv4, pre-ReLU 64 x 28 x 28. Final pooling is 14 x 14. '
        + 'Figshare requires a separate three-class head: 895,747 parameters. '
        + 'This adaptation cannot produce healthy-class predictions. Training must use logits with cross entropy.\n', encoding='utf-8')
    assert total == 895812, f'STOP: parameter discrepancy {total}'
    return rows

if __name__ == '__main__':
    run()
