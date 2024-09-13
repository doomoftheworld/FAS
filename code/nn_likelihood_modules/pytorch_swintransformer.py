from torchvision import models

def get_swin_t(pretrained=False, **kwargs):
    """
    Return the tiny swin transformer model.

    pretrained: Load the pretrained weights or not.
    """
    if pretrained:
        return models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1, **kwargs)
    else:
        return models.swin_t(**kwargs)
    
def get_swin_s(pretrained=False, **kwargs):
    """
    Return the small swin transformer model.

    pretrained: Load the pretrained weights or not.
    """
    if pretrained:
        return models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1, **kwargs)
    else:
        return models.swin_s(**kwargs)

def get_swin_b(pretrained=False, **kwargs):
    """
    Return the basic swin transformer model.

    pretrained: Load the pretrained weights or not.
    """
    if pretrained:
        return models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1, **kwargs)
    else:
        return models.swin_b(**kwargs)
    
def get_swin_v2_t(pretrained=False, **kwargs):
    """
    Return the tiny swin transformer model (version 2).

    pretrained: Load the pretrained weights or not.
    """
    if pretrained:
        return models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1, **kwargs)
    else:
        return models.swin_v2_t(**kwargs)
    
def get_swin_v2_s(pretrained=False, **kwargs):
    """
    Return the small swin transformer model (version 2).

    pretrained: Load the pretrained weights or not.
    """
    if pretrained:
        return models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1, **kwargs)
    else:
        return models.swin_v2_s(**kwargs)
    
def get_swin_v2_b(pretrained=False, **kwargs):
    """
    Return the basic swin transformer model (version 2).

    pretrained: Load the pretrained weights or not.
    """
    if pretrained:
        return models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1, **kwargs)
    else:
        return models.swin_v2_b(**kwargs)