from torch import nn
from torchvision import models
from src.models.brnet import BrNet

def build_model(name,regime='full',pretrained=False,classes=3):
    if classes!=3:raise ValueError('This factory is TRACK_A only')
    if regime not in ['frozen','partial','full']:raise ValueError('Unknown regime')
    if name=='brnet':
        if regime!='full':raise ValueError('BrNet has no pretrained frozen regime')
        return BrNet(classes)
    specs={'mobilenet_v2':(models.mobilenet_v2,models.MobileNet_V2_Weights.IMAGENET1K_V2),
           'efficientnet_b7':(models.efficientnet_b7,models.EfficientNet_B7_Weights.IMAGENET1K_V1),
           'vit_b_16':(models.vit_b_16,models.ViT_B_16_Weights.IMAGENET1K_V1)}
    constructor,weights=specs[name];model=constructor(weights=weights if pretrained else None)
    if name=='vit_b_16':model.heads.head=nn.Linear(model.heads.head.in_features,classes);head='heads'
    else:model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,classes);head='classifier'
    partial={'mobilenet_v2':['features.17','features.18'], 'efficientnet_b7':['features.8'],
             'vit_b_16':['encoder.layers.encoder_layer_10','encoder.layers.encoder_layer_11','encoder.ln']}[name]
    for parameter_name,p in model.named_parameters():
        p.requires_grad=regime=='full' or parameter_name.startswith(head+'.') or (regime=='partial' and any(parameter_name.startswith(prefix+'.') for prefix in partial))
    return model

def training_mode(model):
    model.train()
    for module in model.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in module.parameters()):module.eval()
