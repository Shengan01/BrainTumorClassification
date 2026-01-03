import torch
import torch.nn as nn
from torchvision import models
import timm

def get_baselines(num_classes=4, freeze=True):
    """
    Load pre-trained baselines with proper layer freezing.
    """
    models_dict = {}
    
    # ResNet50 - Freeze all except layer4 and fc
    resnet = models.resnet50(weights='IMAGENET1K_V1')
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    if freeze:
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        for param in resnet.fc.parameters():
            param.requires_grad = True
    models_dict['ResNet'] = resnet
    
    # DenseNet121 - Freeze all except denseblock4 and classifier
    densenet = models.densenet121(weights='IMAGENET1K_V1')
    densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)
    if freeze:
        for param in densenet.parameters():
            param.requires_grad = False
        for name, param in densenet.named_parameters():
            if 'denseblock4' in name or 'classifier' in name or 'norm5' in name:
                param.requires_grad = True
    models_dict['DenseNet121'] = densenet
    
    # ViT - Don't freeze (works well as-is)
    try:
        vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        models_dict['ViT'] = vit
    except Exception as e:
        print(f"Warning: Failed to load ViT: {e}")

    # EfficientNetV2 - Use smaller model with NO freezing (it's already small)
    try:
        # Use the small variant and DON'T freeze - let it train fully
        effnet = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=num_classes)
        # EfficientNet seems to need full training - don't freeze
        models_dict['EfficientNetV2'] = effnet
    except Exception as e:
        print(f"Warning: Failed to load EfficientNetV2: {e}")
        
    # ConvNeXt - Freeze except stages.3 and head
    try:
        convnext = timm.create_model('convnext_base', pretrained=True, num_classes=num_classes)
        if freeze:
            for param in convnext.parameters():
                param.requires_grad = False
            for name, param in convnext.named_parameters():
                if 'stages.3' in name or 'head' in name or 'norm_pre' in name:
                    param.requires_grad = True
        models_dict['ConvNeXt'] = convnext
    except Exception as e:
        print(f"Warning: Failed to load ConvNeXt: {e}")
        
    # RegNetY - Freeze except s4 and head
    try:
        regnet = timm.create_model('regnety_032', pretrained=True, num_classes=num_classes)
        if freeze:
            for param in regnet.parameters():
                param.requires_grad = False
            for name, param in regnet.named_parameters():
                if 's4' in name or 'head' in name:
                    param.requires_grad = True
        models_dict['RegNetY-032'] = regnet
    except Exception as e:
        print(f"Warning: Failed to load RegNetY: {e}")
        
    # Swin Transformer - Don't freeze
    try:
        swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
        models_dict['Swin Transformer'] = swin
    except Exception as e:
        print(f"Warning: Failed to load Swin Transformer: {e}")
    
    # MobileNetV3-Large - Lightweight CNN with depthwise separable convolutions
    # Don't freeze - already small and efficient, trains well end-to-end
    try:
        mobilenet = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=num_classes)
        models_dict['MobileNetV3'] = mobilenet
    except Exception as e:
        print(f"Warning: Failed to load MobileNetV3: {e}")
    
    # MobileViT-S - CNN+Transformer hybrid (most similar to user's Hybrid architecture)
    # Don't freeze - hybrid models train well end-to-end
    try:
        mobilevit = timm.create_model('mobilevit_s', pretrained=True, num_classes=num_classes)
        models_dict['MobileViT'] = mobilevit
    except Exception as e:
        print(f"Warning: Failed to load MobileViT: {e}")

    return models_dict
