# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn as nn
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from transformers.models.qwen2.modeling_qwen2 import (Qwen2Attention, ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward)
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from IPython.core.debugger import set_trace

_use_top_left_mask = flash_attn_supports_top_left_mask()

def FlashAttention2ForPackedTrainingFoward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    is_causal,
    dropout=0.0,
    softmax_scale=None,
    # use_sliding_windows=False,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    use_top_left_mask: bool = False,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`int`, *optional*):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_sliding_windows (`bool`, *optional*):
            Whether to activate sliding window attention.
    """
    assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    cu_seqlens = attention_mask.squeeze(0)

    with torch.no_grad():
        max_seqlen = max([
            cu_seqlens[idx+1] - cu_seqlens[idx]
            for idx in range(cu_seqlens.size(0) - 1)
        ]).item()

    
    causal = is_causal
    
    # Decide whether to use SWA or not by layer index.
    
    if sliding_window is not None and key_states.shape[1] > sliding_window:
        use_sliding_windows = True
    else:
        use_sliding_windows = False
    if not use_sliding_windows:
        attn_output = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    else:
        attn_output = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(sliding_window, sliding_window),
        )

    query_states = query_states.unsqueeze(0)
    key_states = key_states.unsqueeze(0)
    value_states = value_states.unsqueeze(0)
    return attn_output

def FlashAttention2ForPackedTraining(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # This is before the transpose
    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    attn_output = FlashAttention2ForPackedTrainingFoward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=module.is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=target_dtype,
        **kwargs,
    )

    return attn_output, None


    
def replace_qwen2_attention_class():
    ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = FlashAttention2ForPackedTraining
    # Qwen2Attention = Qwen2FlashAttention2ForPackedTraining
    print('Replace QWEN2_ATTENTION_CLASSES to support packed training!!')
