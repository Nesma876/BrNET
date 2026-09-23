import copy
import torch
import torch.nn.functional as F

def logits(model,x):return model.logits(x) if hasattr(model,'logits') else model(x)

def normalize(maps):
    flat=maps.flatten(1);lo=flat.min(1).values[:,None,None];span=(flat.max(1).values-flat.min(1).values)[:,None,None]
    return torch.where(span>0,(maps-lo)/span.clamp_min(1e-12),torch.zeros_like(maps))

def gradcam(model,x,targets=None,layer='features.conv4'):
    module=dict(model.named_modules()).get(layer)
    if module is None:raise ValueError('Missing Grad-CAM layer')
    activation=[];handle=module.register_forward_hook(lambda m,a,o:activation.append(o))
    was_training=model.training;model.eval()
    try:
        x=x.detach().requires_grad_(True);scores=logits(model,x)
        if targets is None:targets=scores.argmax(1)
        targets=torch.as_tensor(targets,device=x.device,dtype=torch.long)
        selected=scores.gather(1,targets[:,None]).sum()
        gradients=torch.autograd.grad(selected,activation[0])[0]
        cam=(gradients.mean((2,3),keepdim=True)*activation[0]).sum(1).relu()
        cam=F.interpolate(cam[:,None],size=x.shape[-2:],mode='bilinear',align_corners=False)[:,0]
        return normalize(cam).detach()
    finally:handle.remove();model.train(was_training)

def integrated_gradients(model,x,targets=None,baseline=None,steps=64):
    if steps<1:raise ValueError('steps must be positive')
    baseline=torch.zeros_like(x) if baseline is None else baseline
    if baseline.shape!=x.shape:raise ValueError('Baseline shape mismatch')
    was_training=model.training;model.eval()
    try:
        with torch.no_grad():
            output=logits(model,x);base_output=logits(model,baseline)
        if targets is None:targets=output.argmax(1)
        targets=torch.as_tensor(targets,device=x.device,dtype=torch.long)
        total=torch.zeros_like(x)
        for j in range(steps+1):
            z=(baseline+(j/steps)*(x-baseline)).detach().requires_grad_(True)
            g=torch.autograd.grad(logits(model,z).gather(1,targets[:,None]).sum(),z)[0]
            total+=g*(.5 if j in (0,steps) else 1.)
        attribution=(x-baseline)*total/steps
        expected=(output-base_output).gather(1,targets[:,None]).flatten()
        delta=attribution.flatten(1).sum(1)-expected
        return attribution.detach(),delta.detach()
    finally:model.train(was_training)

def randomized_copy(model,seed=2026,modules=None):
    clone=copy.deepcopy(model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for name,module in clone.named_modules():
            if hasattr(module,'reset_parameters') and (modules is None or name in modules):module.reset_parameters()
    return clone

def cascading_randomizations(model,seed=2026):
    names=[name for name,module in model.named_modules() if hasattr(module,'reset_parameters')][::-1]
    for end in range(1,len(names)+1):yield names[:end],randomized_copy(model,seed,names[:end])
