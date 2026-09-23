import numpy as np
import pytest
import torch
from torch import nn
from src.models.brnet import BrNet
from src.xai.methods import gradcam,integrated_gradients,randomized_copy
from src.xai.localization import localization,null_maps

def test_gradcam_targets_and_layer_regression():
    torch.set_num_threads(2);m=BrNet(3).eval();x=torch.rand(1,1,256,256)
    shapes=[];handle=m.features.conv4.register_forward_hook(lambda module,args,out:shapes.append(tuple(out.shape)))
    m(x);handle.remove();assert shapes==[(1,64,28,28)]
    for target in [None,[1]]:
        cam=gradcam(m,x,target);assert cam.shape==(1,256,256);assert torch.isfinite(cam).all();assert cam.min()>=0 and cam.max()<=1
    with pytest.raises(ValueError):gradcam(m,x,layer='missing')

def test_ig_linear_completeness():
    model=nn.Sequential(nn.Flatten(),nn.Linear(4,3,bias=False));x=torch.ones(1,1,2,2)
    for target in [None,[2]]:
        attribution,delta=integrated_gradients(model,x,target,steps=8)
        assert attribution.shape==x.shape;assert delta.abs().max()<1e-6

def test_localization_and_nulls():
    mask=np.zeros((8,8),bool);mask[2:4,2:4]=True
    r=localization(mask.astype(float),mask);assert r['saliency_mass']==1 and r['maximum_iou']==1 and r['auprc']==1
    uniform=localization(np.ones((8,8)),mask);assert uniform['saliency_enrichment']==1
    assert uniform['pointing_game']==pytest.approx(mask.mean())
    assert localization(np.zeros((8,8)),mask)['saliency_mass'] is None
    a=null_maps(mask);b=null_maps(mask);np.testing.assert_array_equal(a['random'],b['random'])
    with pytest.raises(ValueError):localization(np.zeros((4,4)),mask)

def test_parameter_randomization_no_mutation():
    m=BrNet(3);original=m.dense.weight.detach().clone();r=randomized_copy(m)
    assert torch.equal(m.dense.weight,original);assert not torch.equal(r.dense.weight,original)
