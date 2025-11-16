# Adaptive S²-Attn Implementation
# Modified from LongLoRA's llama_attn_replace.py

import warnings
from typing import Optional, Tuple

import torch
from torch import nn
import transformers
from einops import rearrange
from flash_attn import __version__ as flash_attn_version
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func
)
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, rotate_half
from flash_attn.bert_padding import unpad_input, pad_input
import math

# Default group size ratio (can be overridden by adaptive predictor)
DEFAULT_GROUP_SIZE_RATIO = 1 / 4


class AdaptiveS2AttnPredictor(nn.Module):
    """
    Predicts adaptive parameters for S²-Attn:
    - group_size_ratio: adaptive group size
    - shift_ratio: adaptive shift amount (default 0.5)
    """

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Lightweight MLP for predicting group size ratio
        self.group_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

        # Lightweight MLP for predicting shift ratio
        self.shift_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

        # Initialize to default values
        self._init_weights()

    def _init_weights(self):
        """Initialize to produce default S²-Attn behavior initially"""
        # Initialize group predictor to output ~0.25 (1/4 ratio)
        nn.init.constant_(self.group_predictor[-2].bias, 0.0)
        nn.init.constant_(self.group_predictor[-2].weight, 0.0)

        # Initialize shift predictor to output ~0.5 (half group shift)
        nn.init.constant_(self.shift_predictor[-2].bias, 0.0)
        nn.init.constant_(self.shift_predictor[-2].weight, 0.0)

    def forward(self, hidden_states, q_len):
        """
        Args:
            hidden_states: [bsz, q_len, hidden_size]
            q_len: sequence length
        Returns:
            group_size: int
            shift_amount: int
        """
        # Use mean pooled features for prediction
        pooled = hidden_states.mean(dim=1)  # [bsz, hidden_size]

        # Predict group size ratio (between 0.1 and 0.5)
        group_ratio = self.group_predictor(pooled).mean()  # Average across batch
        group_ratio = 0.1 + group_ratio * 0.4  # Scale to [0.1, 0.5]

        # Predict shift ratio (between 0.25 and 0.75)
        shift_ratio = self.shift_predictor(pooled).mean()  # Average across batch
        shift_ratio = 0.25 + shift_ratio * 0.5  # Scale to [0.25, 0.75]

        # Calculate actual group size (must be divisible)
        group_size = self._calculate_valid_group_size(q_len, group_ratio)
        shift_amount = int(group_size * shift_ratio)

        return group_size, shift_amount

    def _calculate_valid_group_size(self, q_len, ratio):
        """Ensure group size divides q_len evenly"""
        target_size = int(q_len * ratio)

        # Find closest divisor
        best_size = target_size
        min_diff = float('inf')

        for size in range(max(128, target_size - 512), min(q_len, target_size + 512), 128):
            if q_len % size == 0:
                diff = abs(size - target_size)
                if diff < min_diff:
                    min_diff = diff
                    best_size = size

        # Fallback to default if no good divisor found
        if q_len % best_size != 0:
            best_size = int(q_len * DEFAULT_GROUP_SIZE_RATIO)
            if q_len % best_size != 0:
                best_size = q_len // 4  # Absolute fallback

        return best_size


def forward_flashattn_adaptive(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Adaptive S²-Attn with flash attention"""

    if not self.training:
        warnings.warn("This function should be used just for training. For inference, use forward_flashattn_inference.")

    if output_attentions:
        warnings.warn("Output attentions is not supported, returning None.")

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # === ADAPTIVE PART ===
    # Predict adaptive group size and shift amount
    if hasattr(self, 'adaptive_predictor') and self.adaptive_predictor is not None:
        group_size, shift_amount = self.adaptive_predictor(hidden_states, q_len)
    else:
        # Fallback to default S²-Attn behavior
        group_size = int(q_len * DEFAULT_GROUP_SIZE_RATIO)
        shift_amount = group_size // 2

    if q_len % group_size > 0:
        # Fallback to safe default
        group_size = int(q_len * DEFAULT_GROUP_SIZE_RATIO)
        shift_amount = group_size // 2
        if q_len % group_size > 0:
            raise ValueError(f"q_len {q_len} should be divisible by group size {group_size}")

    # === END ADAPTIVE PART ===

    qkv = torch.stack([query_states, key_states, value_states], dim=2)
    qkv = qkv.transpose(1, 3)

    key_padding_mask = attention_mask.repeat(2, 1)
    nheads = qkv.shape[-2]

    # Apply adaptive shift
    qkv = qkv.reshape(bsz, q_len, 3, 2, self.num_heads // 2, self.head_dim).permute(0, 3, 1, 2, 4, 5).reshape(
        bsz * 2, q_len, 3, self.num_heads // 2, self.head_dim
    )

    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)

    # Use adaptive shift amount instead of fixed group_size // 2
    cu_q_len_tmp = torch.arange(0, max_s, group_size, device=key_padding_mask.device, dtype=cu_q_lens.dtype)
    cu_q_len_tmp = torch.stack([cu_q_len_tmp, cu_q_len_tmp + shift_amount]).repeat(bsz, 1) + cu_q_lens[:-1].unsqueeze(
        -1)
    cu_q_lens = torch.cat([cu_q_len_tmp, cu_q_lens[1:].unsqueeze(-1)], dim=-1).view(-1)

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads // 2)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, group_size, 0.0, softmax_scale=None, causal=True
    )
    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz * 2, q_len),
        "b s (h d) -> b s h d",
        h=nheads // 2,
    )
    output = output.reshape(bsz, 2, q_len, nheads // 2, self.head_dim).transpose(1, 2).reshape(
        bsz, q_len, nheads, self.head_dim
    )

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


def forward_noflashattn_adaptive(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Adaptive S²-Attn without flash attention"""

    bsz, q_len, _ = hidden_states.size()

    # === ADAPTIVE PART ===
    if hasattr(self, 'adaptive_predictor') and self.adaptive_predictor is not None:
        group_size, shift_amount = self.adaptive_predictor(hidden_states, q_len)
    else:
        group_size = int(q_len * DEFAULT_GROUP_SIZE_RATIO)
        shift_amount = group_size // 2

    if q_len % group_size > 0:
        group_size = int(q_len * DEFAULT_GROUP_SIZE_RATIO)
        shift_amount = group_size // 2
        if q_len % group_size > 0:
            raise ValueError(f"q_len {q_len} should be divisible by group size {group_size}")

    num_group = q_len // group_size
    # === END ADAPTIVE PART ===

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Adaptive shift function
    def shift_adaptive(qkv, bsz, q_len, group_size, num_heads, head_dim, shift_amount):
        qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-shift_amount, dims=2)
        qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
        return qkv

    query_states = shift_adaptive(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim, shift_amount)
    key_states = shift_adaptive(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim, shift_amount)
    value_states = shift_adaptive(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim, shift_amount)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz * num_group, self.num_heads, group_size, group_size):
        raise ValueError(
            f"Attention weights should be of size {(bsz * num_group, self.num_heads, group_size, group_size)}, but is"
            f" {attn_weights.size()}"
        )

    attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)
    if attention_mask is not None:
        if attention_mask.size() != (bsz * num_group, 1, group_size, group_size):
            raise ValueError(
                f"Attention mask should be of size {(bsz * num_group, 1, group_size, group_size)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz * num_group, self.num_heads, group_size, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * num_group, self.num_heads, group_size, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    # Shift back with adaptive amount
    attn_output[:, :, self.num_heads // 2:] = attn_output[:, :, self.num_heads // 2:].roll(shift_amount, dims=1)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# Keep the same inference and helper functions
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    return attention_mask


def apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids):
    gather_indices = position_ids[:, :, None, None]
    gather_indices = gather_indices.repeat(1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3])
    bsz = gather_indices.shape[0]
    cos, sin = (
        torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
        for x in cos_sin
    )
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    return q, k


def forward_flashattn_inference(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Standard full attention for inference - NO adaptive changes needed"""
    if output_attentions:
        warnings.warn("Output attentions is not supported, returning None.")

    bsz, q_len, _ = hidden_states.size()
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

    q, k, v = (
        op(hidden_states).view(bsz, q_len, nh, self.head_dim)
        for op, nh in (
        (self.q_proj, self.num_heads),
        (self.k_proj, kv_heads),
        (self.v_proj, kv_heads),
    )
    )

    kv_seq_len = k.shape[1]
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[2]
        kv_seq_len += past_kv_len

    cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
    q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)

    if past_key_value is not None:
        assert flash_attn_version >= "2.1.0", "past_key_value support requires flash-attn >= 2.1.0"
        k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
        v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)

    past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

    if attention_mask is None:
        output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True).view(bsz, q_len, -1)
    else:
        q, indices, cu_q_lens, max_s = unpad_input(q, attention_mask[:, -q_len:])
        kv, _, cu_k_lens, max_k = unpad_input(torch.stack((k, v), dim=2), attention_mask)
        output_unpad = flash_attn_varlen_kvpacked_func(
            q, kv, cu_q_lens, cu_k_lens, max_s, max_k, 0.0, softmax_scale=None, causal=True
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)

    return self.o_proj(output), None, past_key_value


def _prepare_decoder_attention_mask_inference(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )
    if attention_mask is not None and torch.all(attention_mask):
        return None
    return attention_mask


def replace_llama_attn_adaptive(use_flash_attn=True, use_full=False, inference=False, use_adaptive=True):
    """
    Replace LLaMA attention with Adaptive S²-Attn

    Args:
        use_flash_attn: Whether to use flash attention
        use_full: Whether to use full attention (no S²-Attn)
        inference: Whether in inference mode
        use_adaptive: Whether to use adaptive S²-Attn (new parameter)
    """
    if use_flash_attn:
        cuda_major, cuda_minor = torch.cuda.get_device_capability()
        if cuda_major < 8:
            warnings.warn(
                "Flash attention is only supported on A100 or H100 GPU during training."
            )

        if inference:
            # Inference always uses full attention
            transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
                _prepare_decoder_attention_mask_inference
            )
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_inference
        else:
            # Training: use adaptive if requested
            transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
                _prepare_decoder_attention_mask
            )
            if use_full:
                # Import from original file
                from llama_attn_replace import forward_flashattn_full
                transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_full
            elif use_adaptive:
                transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_adaptive
            else:
                # Use original S²-Attn
                from llama_attn_replace import forward_flashattn
                transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn
    else:
        if use_adaptive and not use_full and not inference:
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_noflashattn_adaptive
        else:
            from llama_attn_replace import forward_noflashattn
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_noflashattn


def add_adaptive_predictor_to_model(model, hidden_size=None, num_heads=None):
    """
    Add adaptive predictor to all attention layers

    Args:
        model: The transformer model
        hidden_size: Hidden size (auto-detected if None)
        num_heads: Number of attention heads (auto-detected if None)
    """
    if hidden_size is None:
        hidden_size = model.config.hidden_size
    if num_heads is None:
        num_heads = model.config.num_attention_heads

    # Add predictor to each attention layer
    for name, module in model.named_modules():
        # --- THIS IS THE FIX ---
        # The 'self_attn' in name check was causing the RecursionError
        if isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention):
            module.adaptive_predictor = AdaptiveS2AttnPredictor(hidden_size, num_heads)
            print(f"Added adaptive predictor to {name}")

    return model
