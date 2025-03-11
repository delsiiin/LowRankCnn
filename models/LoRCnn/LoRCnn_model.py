import copy
import json
import time

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig

from transformers import Trainer, BitsAndBytesConfig


from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download

from .LoRCnn_config import LoRCnnConfig
from .utils import CausalConv2d, UpsampleFP32, KeepRes, ChannelSplit
import torch.nn.functional as F

# from transformers import LlamaForCausalLM

from .modeling_llama_cnn import LlamaForCausalLM as BaseLlamaForCausalLM
from .modeling_llama_cnn import LlamaAttention as BaseLlamaAttention
from .modeling_llama_cnn import LlamaDecoderLayer as BaseLlamaDecoderLayer
from .modeling_llama_cnn import apply_rotary_pos_emb
from transformers.models.llama.configuration_llama import LlamaConfig

class LoRCnnAttention(nn.Module):

    def __init__(self, config: LoRCnnConfig, idx, base_model):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = base_model.model.layers[idx].self_attn.q_proj
        self.k_proj = base_model.model.layers[idx].self_attn.k_proj
        self.v_proj = base_model.model.layers[idx].self_attn.v_proj
        self.o_proj = base_model.model.layers[idx].self_attn.o_proj
        self.rotary_emb = base_model.model.layers[idx].self_attn.rotary_emb

        self.attn_down_proj_dim = self.config.attn_down_proj_dim // self.config.num_attention_heads
        self.num_deep_cnn_layers = self.config.num_deep_cnn_layers
        self.inner_channel = self.config.inner_channel

        self.attn_down_proj_q = nn.Linear(self.config.hidden_size // self.config.num_attention_heads, self.attn_down_proj_dim, bias=False)
        self.attn_down_proj_k = nn.Linear(self.config.hidden_size // self.config.num_attention_heads, self.attn_down_proj_dim, bias=False) 

        self.attention_predictor_cnn = nn.Sequential(
                nn.LayerNorm(self.config.max_position_embeddings),
                *[
                    deep_layer
                    for _ in range(self.num_deep_cnn_layers)
                    for deep_layer in (
                        CausalConv2d(self.inner_channel*self.config.num_attention_heads, self.inner_channel*self.config.num_attention_heads, (63,1), padding=(63 // 2 * 1, 0), dilation=(1, 1), stride=(1,1), groups=self.config.num_attention_heads),
                        nn.ReLU(),
                    )
                ],
                nn.LayerNorm(self.config.max_position_embeddings)
            ) 

        self.attention_predictor_dec_scaler = nn.Linear(self.config.max_position_embeddings, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # <<< LoRCnn >>>
        query_states = self.attn_down_proj_q(query_states)
        key_states = self.attn_down_proj_k(key_states)
        # <<< LoRCnn >>>

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # <<< LoRCnn >>>
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.attn_down_proj_dim)
        estimated_scale = self.attention_predictor_dec_scaler(attn_weights)
        # <<< LoRCnn >>>

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # <<< LoRCnn >>>
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = self.attention_predictor_cnn(attn_weights)
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        # <<< LoRCnn >>>

        # upcast attention to fp32
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # <<< LoRCnn >>>
        if self.training:
            with torch.autocast('cuda', torch.float32):
                # return DUMMY_OUTPUT #1778
                # attn_weights = F.log_softmax(attn_weights.to(torch.float32), dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights = attn_weights * torch.sigmoid(estimated_scale)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = attn_weights * torch.sigmoid(estimated_scale)
        # <<< LoRCnn >>>

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
class LoRCnnLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LoRCnnConfig, idx, base_model):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = LoRCnnAttention(config, idx, base_model)
        self.mlp = base_model.model.layers[idx].mlp
        self.input_layernorm = base_model.model.layers[idx].input_layernorm
        self.post_attention_layernorm = base_model.model.layers[idx].post_attention_layernorm 
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # <<< LoRCnn >>>
        attn_output = hidden_states
        # <<< LoRCnn >>>

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            # <<< LoRCnn >>>
            outputs += (self_attn_weights, attn_output,)
            # <<< LoRCnn >>>

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class LoRCnnModel(nn.Module):

    def __init__(
            self,
            base_model,
            LoRCnn_config,
    ):

        super().__init__()
        self.base_model = base_model

        self.config = LoRCnn_config

        self.LoRCNN_layers = nn.ModuleList([LoRCnnLlamaDecoderLayer(LoRCnn_config, idx, base_model) for idx in range(LoRCnn_config.num_hidden_layers)])

        self.device = base_model.model.layers[-1].self_attn.q_proj.weight.device

        self.dtype = self.base_model.dtype

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._require_grads_hook = self.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def disable_input_require_grads(self):
        """
        Removes the `_require_grads_hook`.
        """
        self._require_grads_hook.remove()

    @classmethod
    def from_pretrained(
            cls,
            base_model_path=None,
            LoRCnn_model_path=None,
            Type="LLaMA",
            load_in_4bit=False,
            load_in_8bit=False,
            **kwargs,
    ):
        
        LoRCnn_config = LoRCnnConfig.from_pretrained(LoRCnn_model_path)
            
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        base_model = BaseLlamaForCausalLM.from_pretrained(
            base_model_path, 
            quantization_config=quantization_config if load_in_4bit else None,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            **kwargs
        )

        model = cls(
            base_model,
            LoRCnn_config
        )

        LoRCnn_attn_down_proj_qs_path = os.path.join(LoRCnn_model_path, "attn_down_proj_qs.pt")
        if os.path.exists(LoRCnn_attn_down_proj_qs_path):
            filename = LoRCnn_attn_down_proj_qs_path
        else:
            filename = hf_hub_download(LoRCnn_model_path, "ssd_model.pt")
        LoRCnn_state_dict = torch.load(filename, map_location=base_model.device)
        model.attn_down_proj_qs.load_state_dict(LoRCnn_state_dict, strict=False)

        LoRCnn_attn_down_proj_ks_path = os.path.join(LoRCnn_model_path, "attn_down_proj_ks.pt")
        if os.path.exists(LoRCnn_attn_down_proj_ks_path):
            filename = LoRCnn_attn_down_proj_ks_path
        else:
            filename = hf_hub_download(LoRCnn_model_path, "ssd_model.pt")
        LoRCnn_state_dict = torch.load(filename, map_location=base_model.device)
        model.attn_down_proj_ks.load_state_dict(LoRCnn_state_dict, strict=False)

        LoRCnn_attention_predictor_cnns_path = os.path.join(LoRCnn_model_path, "attention_predictor_cnns.pt")
        if os.path.exists(LoRCnn_attention_predictor_cnns_path):
            filename = LoRCnn_attention_predictor_cnns_path
        else:
            filename = hf_hub_download(LoRCnn_model_path, "ssd_model.pt")
        LoRCnn_state_dict = torch.load(filename, map_location=base_model.device)
        model.attention_predictor_cnns.load_state_dict(LoRCnn_state_dict, strict=False)

        LoRCnn_attention_predictor_dec_scalers_path = os.path.join(LoRCnn_model_path, "attention_predictor_dec_scalers.pt")
        if os.path.exists(LoRCnn_attention_predictor_dec_scalers_path):
            filename = LoRCnn_attention_predictor_dec_scalers_path
        else:
            filename = hf_hub_download(LoRCnn_model_path, "ssd_model.pt")
        LoRCnn_state_dict = torch.load(filename, map_location=base_model.device)
        model.attention_predictor_dec_scalers.load_state_dict(LoRCnn_state_dict, strict=False)

        return model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple]:
        
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.base_model.model.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self.base_model.model._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        hidden_states_base = inputs_embeds.detach()
    
        # if self.gradient_checkpointing and self.training:
        #     if use_cache:
        #         logger.warning_once(
        #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        #         )
        #         use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        if self.training:
            loss_attn_weights = 0
            loss_attn_output = 0
            loss_per_layer = 0

        for idx, decoder_layer in enumerate(self.base_model.model.layers):
            # if output_hidden_states:
            #     all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.training:

                with torch.no_grad():
                    layer_outputs_base = decoder_layer(
                        hidden_states_base,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=True,
                        use_cache=use_cache,
                    )

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, True, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.LoRCNN_layers[idx]),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )

            else:
                
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cnn_mode=True,
                    attn_down_proj_dim=self.attn_down_proj_dim,
                    down_proj_q_per_layer=self.attn_down_proj_qs[idx],
                    down_proj_k_per_layer=self.attn_down_proj_ks[idx],
                    attention_predictor_cnn_per_layer=self.attention_predictor_cnns[idx],
                    attention_predictor_dec_scaler_per_layer=self.attention_predictor_dec_scalers[idx],
                )

            
            if self.training:
                # hidden_states_base = layer_outputs_base[0].detach()
                # hidden_states = layer_outputs[0]

                # attn_weights_base = layer_outputs_base[1].detach()
                # attn_weights = layer_outputs[1]

                # attn_output_base = layer_outputs_base[2]
                # attn_output = layer_outputs[2]

                hidden_states_base = layer_outputs_base[0].detach()
                hidden_states = layer_outputs[0]

                attn_weights_base = layer_outputs_base[1].detach()
                attn_weights = layer_outputs[1]

                attn_output_base = layer_outputs_base[2].detach()
                attn_output = layer_outputs[2]

                with torch.autocast('cuda', torch.float32):
                    loss_attn_weights += F.kl_div(
                        F.log_softmax(attn_weights.to(torch.float32).view(-1, attn_weights.shape[-1]), dim=-1, dtype=torch.float32),
                        F.softmax(attn_weights_base.to(torch.float32).view(-1, attn_weights_base.shape[-1]), dim=-1, dtype=torch.float32),
                        reduction='batchmean',
                    ) * 0.1
                    # return DUMMY_OUTPUT #2738
                    loss_attn_weights += F.mse_loss(
                        F.softmax(attn_weights.to(torch.float32).view(-1, attn_weights.shape[-1]), dim=-1, dtype=torch.float32), 
                        F.softmax(attn_weights_base.to(torch.float32).view(-1, attn_weights_base.shape[-1]), dim=-1, dtype=torch.float32),
                    )
                    # del attn_weights
                    # del attn_weights_base

                # print(loss_attn_weights, "loss_attn_weights")

                loss_attn_output += F.mse_loss(
                    attn_output, 
                    attn_output_base
                )
                # # del attn_output
                # # del attn_output_base

                # print(loss_attn_output, "loss_attn_output")

                loss_per_layer += F.mse_loss(hidden_states.to(torch.float32), hidden_states_base.to(torch.float32))

                # print(loss_per_layer, "loss_per_layer")

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            # if output_attentions:
            #     all_self_attns += (layer_outputs[1],)
        if self.training:
            total_loss_avg = (loss_per_layer + loss_attn_output + loss_attn_weights) / len(self.base_model.model.layers)

            # print(total_loss_avg, "total_loss_avg")

            # hidden_states_base = self.base_model.model.norm(hidden_states_base)
        hidden_states = self.base_model.model.norm(hidden_states)

        # add hidden states from the last decoder layer
        # if output_hidden_states:
        #     all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        outputs = tuple(v for v in [hidden_states_base, next_cache, all_hidden_states, all_self_attns, hidden_states, total_loss_avg] if v is not None)
        
        hidden_states_base = outputs[0].detach()
        logits_base = self.base_model.lm_head(hidden_states_base).detach()

        final_loss = None

        if self.training:
            hidden_states = outputs[-2]
            logits = self.base_model.lm_head(hidden_states)

            # del hidden_states_base

            loss_kd = F.kl_div(
                F.log_softmax(logits, dim=-1, dtype=torch.float32), 
                F.softmax(logits_base, dim=-1, dtype=torch.float32),
                reduction='batchmean',
            ) * 0.2

            # print(loss_kd, "loss_kd")

            # # del logits_base

            final_loss = total_loss_avg + loss_kd

            # print(final_loss, "final_loss")
        
        return_dict = False
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (final_loss,) + output if final_loss is not None else output
        