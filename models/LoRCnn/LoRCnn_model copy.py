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

class LoRCnnModel(nn.Module):

    def __init__(
            self,
            base_model,
            LoRCnn_config,
    ):

        super().__init__()
        self.base_model = base_model

        self.config = LoRCnn_config

        self.attn_down_proj_dim = self.config.attn_down_proj_dim // self.config.num_attention_heads
        self.num_deep_cnn_layers = self.config.num_deep_cnn_layers
        self.inner_channel = self.config.inner_channel

        self.attn_down_proj_qs = nn.ModuleList([nn.Linear(self.config.hidden_size // self.config.num_attention_heads, self.attn_down_proj_dim, bias=False) for _ in range(self.config.num_hidden_layers)])
        self.attn_down_proj_ks = nn.ModuleList([nn.Linear(self.config.hidden_size // self.config.num_attention_heads, self.attn_down_proj_dim, bias=False) for _ in range(self.config.num_hidden_layers)])

        self.attention_predictor_cnns = nn.ModuleList([
            nn.Sequential(
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
            ) for _ in range(self.config.num_hidden_layers)
        ])

        self.attention_predictor_dec_scalers = nn.ModuleList([
            nn.Linear(self.config.max_position_embeddings, 1) for _ in range(self.config.num_hidden_layers)
        ])

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

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Module`: A torch module mapping vocabulary to hidden states.
        """
        base_model = getattr(self.base_model, self.base_model.base_model_prefix, self.base_model)
        if base_model is not self.base_model:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

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

    def forward_model(
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

        hidden_states_base = inputs_embeds

        hidden_states = inputs_embeds.clone()

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

                layer_outputs_base = decoder_layer(
                        hidden_states_base,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=True,
                        use_cache=use_cache,
                        cnn_mode=False,
                    )

            if self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, True, None, True, self.attn_down_proj_dim, self.attn_down_proj_qs[idx], self.attn_down_proj_ks[idx], self.attention_predictor_cnns[idx], self.attention_predictor_dec_scalers[idx])

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
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
                hidden_states_base = layer_outputs_base[0]
            hidden_states = layer_outputs[0]

            if self.training:
                attn_weights_base = layer_outputs_base[1]
                attn_weights = layer_outputs[1]

                # if idx < 5:
                #     print(attn_weights, "attn_weights")
                #     print(attn_weights_base, "attn_weights_base")

                attn_output_base = layer_outputs_base[2]
                attn_output = layer_outputs[2]

                with torch.autocast('cuda', torch.float32):
                    loss_attn_weights += F.kl_div(
                        F.log_softmax(attn_weights.to(torch.float32).view(-1, attn_weights.shape[-1]), dim=-1, dtype=torch.float32),
                        F.softmax(attn_weights_base.to(torch.float32).view(-1, attn_weights_base.shape[-1]), dim=-1, dtype=torch.float32),
                        reduction='batchmean',
                    ) * 0.1
                    # return DUMMY_OUTPUT #2738
                    loss_attn_weights += F.mse_loss(
                        attn_weights.to(torch.float32).view(-1, attn_weights_base.shape[-1]), 
                        attn_weights_base.to(torch.float32).view(-1, attn_weights_base.shape[-1]),
                    )
                    # del attn_weights
                    del attn_weights_base

                print(loss_attn_weights, "loss_attn_weights")

                loss_attn_output += F.mse_loss(
                    attn_output_base, 
                    attn_output
                )
                # del attn_output
                del attn_output_base

                print(loss_attn_output, "loss_attn_output")

                loss_per_layer += F.mse_loss(hidden_states.to(torch.float32), hidden_states_base.to(torch.float32))

                print(loss_per_layer, "loss_per_layer")

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            # if output_attentions:
            #     all_self_attns += (layer_outputs[1],)
        if self.training:
            total_loss_avg = (loss_per_layer + loss_attn_output + loss_attn_weights) / len(self.base_model.model.layers)

            print(total_loss_avg, "total_loss_avg")

            hidden_states_base = self.base_model.model.norm(hidden_states_base)
        hidden_states = self.base_model.model.norm(hidden_states)

        # add hidden states from the last decoder layer
        # if output_hidden_states:
        #     all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        return_dict = False
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, hidden_states_base, total_loss_avg] if v is not None)
        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
    
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.forward_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.base_model.lm_head(hidden_states)

        final_loss = None

        if self.training:
            hidden_states_base = outputs[-2]
            logits_base = self.base_model.lm_head(hidden_states_base)

            del hidden_states_base

            loss_kd = F.kl_div(
                F.log_softmax(logits, dim=-1, dtype=torch.float32), 
                F.softmax(logits_base, dim=-1, dtype=torch.float32),
                reduction='batchmean',
            ) * 0.2

            print(loss_kd, "loss_kd")

            del logits_base

            final_loss = outputs[-1] + loss_kd

            print(final_loss, "final_loss")
        
        return_dict = False
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (final_loss,) + output if final_loss is not None else output