# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

def apply_liger_kernel_to_internvit() -> None:
    from livestar.model.livestar_chat import modeling_livestar_vit
    from liger_kernel.transformers.layer_norm import LigerLayerNorm
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    modeling_livestar_vit.NORM2FN['rms_norm'] = LigerRMSNorm
    modeling_livestar_vit.NORM2FN['layer_norm'] = LigerLayerNorm
    print('Liger kernel applied to InternViT')
