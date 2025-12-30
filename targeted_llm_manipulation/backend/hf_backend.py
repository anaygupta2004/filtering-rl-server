import re
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as f
from peft.config import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, BitsAndBytesConfig

from targeted_llm_manipulation.backend.backend import Backend


class HFBackend(Backend):
    def __init__(
        self,
        model_name: str,
        lora_path: Optional[str],
        device: str,
        inference_quantization: Optional[str] = None,
        enable_scratchpad_prefill: bool = True,
        **kwargs,
    ):
        self.device = device
        self.model_name = model_name
        self.enable_scratchpad_prefill = enable_scratchpad_prefill
        assert self.device is not None, "Device must be specified"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.lora_active = False

        if inference_quantization == "8-bit" or inference_quantization == "4-bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=inference_quantization == "8-bit",
                load_in_4bit=inference_quantization == "4-bit",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        self.lora = lora_path is not None

        if self.lora:
            self.model.load_adapter(lora_path, adapter_name="agent")
            config = PeftConfig.from_pretrained(lora_path)
            self.model.add_adapter(config, "environment")
            self.model.set_adapter("environment")
            self.lora_active = False

        if self.tokenizer.pad_token is None:
            pad = "<|finetune_right_pad_id|>" if "Llama-3.1" in model_name else "<|reserved_special_token_198|>"
            self.pad_id = self.tokenizer.convert_tokens_to_ids(pad)
            self.tokenizer.pad_token = pad
            self.tokenizer.pad_token_id = self.pad_id
            self.model.config.pad_token_id = self.pad_id
            self.model.generation_config.pad_token_id = self.pad_id
        else:
            self.pad_id = self.tokenizer.pad_token_id
            self.model.config.pad_token_id = self.pad_id
            self.model.generation_config.pad_token_id = self.pad_id

    @torch.no_grad()
    def get_response(self, messages_in: List[Dict[str, str]], temperature=1, max_tokens=1024, role=None) -> str:
        return self.get_response_vec([messages_in], temperature, max_tokens, role=role)[0]

    @torch.no_grad()
    def get_response_vec(
        self,
        messages_in: List[List[Dict[str, str]]],
        temperature=1,
        max_tokens=1024,
        role: Optional[str] = None,
    ) -> List[str]:
        self.set_lora(role)

        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "pad_token_id": self.pad_id,
            "do_sample": True,
            "use_cache": True,
            "top_k": 0,
        }
        
        model_type = self.model.config.model_type
        if "gemma" in model_type:
            messages_in = [self.fix_messages_for_gemma(messages) for messages in messages_in]

        chat_text = self.tokenizer.apply_chat_template(
            messages_in,
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        assert type(chat_text) is BatchEncoding, "chat_text is not a tensor"
        
        # PRE-FILL with <scratchpad> for Gemma agent to force scratchpad usage
        scratchpad_prefix = "<scratchpad>\n"
        if self.enable_scratchpad_prefill and role == "agent" and "gemma" in model_type:
            prefix_ids = self.tokenizer.encode(scratchpad_prefix, add_special_tokens=False, return_tensors="pt")
            batch_size = chat_text["input_ids"].shape[0]
            prefix_ids = prefix_ids.expand(batch_size, -1)
            chat_text["input_ids"] = torch.cat([chat_text["input_ids"], prefix_ids], dim=1)
            chat_text["attention_mask"] = torch.cat([
                chat_text["attention_mask"], 
                torch.ones(batch_size, prefix_ids.shape[1], dtype=torch.long)
            ], dim=1)
        
        chat_text = chat_text.to(self.device)
        output = self.model.generate(**chat_text, **generation_config).to("cpu")

        if "llama" in model_type:
            assistant_token_id = self.tokenizer.encode("<|end_header_id|>")[-1]
        elif "gemma" in model_type:
            assistant_token_id = self.tokenizer.encode("model")[-1]
        elif "qwen" in model_type:
            assistant_token_id = self.tokenizer.encode("assistant")[-1]
        else:
            assistant_token_id = None

        if assistant_token_id is not None:
            matches = (output == assistant_token_id).nonzero(as_tuple=True)
            if len(matches[1]) > 0:
                start_idx = matches[1][-1]
                if "gemma" in model_type:
                    start_idx += 1
                elif "qwen" in model_type:
                    start_idx += 1
            else:
                start_idx = chat_text["input_ids"].shape[1]
        else:
            start_idx = chat_text["input_ids"].shape[1]

        new_tokens = output[:, start_idx:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        decoded = [m.strip() for m in decoded]
        
        # For Gemma with scratchpad prefill: prepend prefix and clean duplicates
        if self.enable_scratchpad_prefill and role == "agent" and "gemma" in model_type:
            cleaned = []
            for m in decoded:
                # Prefix already in output from prefill, just clean any model duplicates
                # Clean duplicate scratchpad tags
                m = re.sub(r'<scratchpad>\s*<scratchpad>', '<scratchpad>', m)
                cleaned.append(m)
            decoded = cleaned
        
        return decoded

    @torch.no_grad()
    def get_next_token_probs_normalized(self, messages: List[dict], valid_tokens: List[str], role=None) -> dict:
        return self.get_next_token_probs_normalized_vec([messages], [valid_tokens], role=role)[0]

    def aggregate_token_probabilities(self, top_probs, top_indices):
        top_tokens = []
        for probs, indices in zip(top_probs, top_indices):
            token_dict = defaultdict(float)
            for token_index, token_prob in zip(indices, probs):
                token_index = int(token_index)
                token = self.tokenizer.decode([token_index]).lower().strip()
                token_dict[token] += token_prob.item()
            top_tokens.append(dict(token_dict))
        return top_tokens

    @torch.no_grad()
    def get_next_token_probs_normalized_vec(
        self, messages_batch: List[List[dict]], valid_tokens_n: List[List[str]], role=None
    ) -> List[Dict[str, float]]:
        self.set_lora(role)

        if "gemma" in self.model.config.model_type:
            messages_batch = [self.fix_messages_for_gemma(messages) for messages in messages_batch]

        inputs = [
            str(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            + "The answer is: "
            for messages in messages_batch
        ]

        tokenized = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)

        generation_config = {"max_new_tokens": 1, "pad_token_id": self.pad_id, "top_k": 0}

        outputs = self.model.generate(**tokenized, **generation_config, return_dict_in_generate=True, output_scores=True)
        logits_batch = outputs.scores[0]
        probs_batch = f.softmax(logits_batch, dim=-1)

        top_k = 10
        top_probs, top_indices = torch.topk(probs_batch, top_k, dim=-1)
        top_tokens = self.aggregate_token_probabilities(top_probs.to("cpu"), top_indices.to("cpu"))
        
        results = []
        for batch_idx, valid_tokens in enumerate(valid_tokens_n):
            assert len(valid_tokens) > 0, "No valid tokens provided"
            token_prob_dict = top_tokens[batch_idx]
            result = {k: token_prob_dict[k] if k in token_prob_dict else 0 for k in valid_tokens}
            total_prob = sum(result.values())
            result = {k: v / total_prob if total_prob > 0 else 0 for k, v in result.items()}
            results.append(result)
        return results

    @torch.no_grad()
    def set_lora(self, role: Optional[str]):
        if self.lora:
            if role is None or role == "environment":
                self.lora_active = False
                self.model.set_adapter("environment")
            elif role == "agent":
                self.lora_active = True
                self.model.set_adapter("agent")
            else:
                raise ValueError(f"Unsupported role: {role}")

    def close(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

    @staticmethod
    def strip_thinking(text: str) -> str:
        stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return stripped.strip()

    @staticmethod
    def fix_messages_for_gemma(messages_in):
        if not messages_in:
            return []
        messages_in = [m.copy() for m in messages_in]
        if messages_in[0]["role"] == "system":
            if len(messages_in) > 1:
                new_content = f"<Instructions>\n\n {messages_in[0]['content']}</Instructions>\n\n{messages_in[1]['content']}\n\n"
                messages_in[1]["content"] = new_content
                messages_in = messages_in[1:]
            else:
                messages_in[0]["role"] = "user"
        result = []
        for message in messages_in:
            message = message.copy()
            if message["role"] == "function_call":
                message["role"] = "assistant"
            elif message["role"] == "ipython":
                message["role"] = "user"
            elif message["role"] == "system":
                message["role"] = "user"
            if result and result[-1]["role"] == message["role"]:
                result[-1]["content"] += "\n\n" + message["content"]
            else:
                result.append(message)
        if result and result[0]["role"] != "user":
            result.insert(0, {"role": "user", "content": "Continue."})
        return result


    @torch.no_grad()
    def get_response_with_activations(
        self,
        messages_in: List[Dict[str, str]],
        temperature: float = 1,
        max_tokens: int = 1024,
        role: Optional[str] = None,
        layers_to_extract: Optional[List[int]] = None,
    ) -> Tuple[str, Optional[Dict[int, torch.Tensor]]]:
        """
        Generate response and optionally extract activations from specified layers.
        
        Args:
            messages_in: Input messages
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            role: Model role (agent/environment)
            layers_to_extract: List of layer indices to extract activations from.
                             If None, no activations are extracted.
        
        Returns:
            Tuple of (response_text, activations_dict)
            activations_dict is {layer_idx: activations} where activations are [hidden_dim]
            for the last token position. None if layers_to_extract is None.
        """
        if layers_to_extract is None:
            # Fast path - no activation extraction
            response = self.get_response(messages_in, temperature, max_tokens, role)
            return response, None
        
        self.set_lora(role)
        
        model_type = self.model.config.model_type
        if "gemma" in model_type:
            messages_in = self.fix_messages_for_gemma(messages_in)
        
        # Tokenize input
        chat_text = self.tokenizer.apply_chat_template(
            [messages_in],
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(self.device)
        
        # Pre-fill scratchpad if enabled
        scratchpad_prefix = "<scratchpad>\n"
        if self.enable_scratchpad_prefill and role == "agent" and "gemma" in model_type:
            prefix_ids = self.tokenizer.encode(scratchpad_prefix, add_special_tokens=False, return_tensors="pt")
            chat_text["input_ids"] = torch.cat([chat_text["input_ids"], prefix_ids.to(self.device)], dim=1)
            chat_text["attention_mask"] = torch.cat([
                chat_text["attention_mask"],
                torch.ones(1, prefix_ids.shape[1], dtype=torch.long, device=self.device)
            ], dim=1)
        
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "pad_token_id": self.pad_id,
            "do_sample": True,
            "use_cache": True,
            "top_k": 0,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
        }
        
        # Generate with hidden states
        outputs = self.model.generate(**chat_text, **generation_config)
        
        # Extract activations from last generated token
        activations_dict = {}
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            # hidden_states is tuple of (num_generated_tokens, ) where each is tuple of (num_layers, batch, seq, hidden)
            # We want the last token's activations
            last_token_hidden_states = outputs.hidden_states[-1]  # Last generated token
            for layer_idx in layers_to_extract:
                if layer_idx < len(last_token_hidden_states):
                    # Get activations: [batch, seq, hidden] -> take last position
                    layer_activations = last_token_hidden_states[layer_idx][0, -1, :]  # [hidden_dim]
                    activations_dict[layer_idx] = layer_activations.cpu()
        
        # Decode response
        output_ids = outputs.sequences
        if "llama" in model_type:
            assistant_token_id = self.tokenizer.encode("<|end_header_id|>")[-1]
        elif "gemma" in model_type:
            assistant_token_id = self.tokenizer.encode("model")[-1]
        elif "qwen" in model_type:
            assistant_token_id = self.tokenizer.encode("assistant")[-1]
        else:
            assistant_token_id = None
        
        if assistant_token_id is not None:
            matches = (output_ids == assistant_token_id).nonzero(as_tuple=True)
            if len(matches[1]) > 0:
                start_idx = matches[1][-1]
                if "gemma" in model_type or "qwen" in model_type:
                    start_idx += 1
            else:
                start_idx = chat_text["input_ids"].shape[1]
        else:
            start_idx = chat_text["input_ids"].shape[1]
        
        new_tokens = output_ids[:, start_idx:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        response = decoded[0].strip()
        
        # Clean scratchpad duplicates if needed
        if self.enable_scratchpad_prefill and role == "agent" and "gemma" in model_type:
            response = re.sub(r'<scratchpad>\s*<scratchpad>', '<scratchpad>', response)
        
        return response, activations_dict if activations_dict else None
