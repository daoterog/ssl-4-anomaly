from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer


@register_model
def vit_tiny(patch_size=16, **kwargs):
    """ViT-Tiny (Vit-Ti/16)"""
    model_kwargs = dict(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, **kwargs
    )
    model = _create_vision_transformer("vit_tiny_patch16_224", **model_kwargs)
    return model


@register_model
def vit_small(patch_size=16, **kwargs):
    model_kwargs = dict(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, **kwargs
    )
    model = _create_vision_transformer("vit_small_patch16_224", **model_kwargs)
    return model


@register_model
def vit_base(patch_size=16, **kwargs):
    model_kwargs = dict(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, **kwargs
    )
    model = _create_vision_transformer("vit_base_patch16_224", **model_kwargs)
    return model


@register_model
def vit_large(patch_size=16, **kwargs):
    model_kwargs = dict(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )
    model = _create_vision_transformer("vit_large_patch16_224", **model_kwargs)
    return model
