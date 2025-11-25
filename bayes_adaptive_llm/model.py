"""
Lightweight Bayes-Adaptive LLM model skeleton.
Matches the interface of TRIPModel so the trainer/pipeline can be reused,
and exposes a placeholder hook for MCTS-based preference scoring.
"""

from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file as safe_load_file
from peft import PeftModel
import re

from base.model import Model


class BayesAdaptiveLLMModel(Model):
    """
    Classification-style policy head on top of a PLM.
    """
    def __init__(
        self,
        model_name: str = "llama3",
        input_max_len: int = 512,
        stop_symbol: str = "\n",
        cuda: bool = True,
        trust_remote_code: bool = False,
        model_kwargs: Optional[Dict] = None,
    ):
        
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.cuda = self.device.type == "cuda"
        load_kwargs = model_kwargs.copy() if model_kwargs else {}
        base_model_name = load_kwargs.pop("base_model_name_or_path", None)
        adapter_path = Path(model_name)
        is_adapter = adapter_path.exists() and (adapter_path / "adapter_config.json").exists()
        if is_adapter:
            if base_model_name is None:
                raise ValueError(
                    "Detected a PEFT/LoRA adapter at %s but no base model was provided. "
                    "Set --local-base-model when running GDPZero." % model_name
                )
            tokenizer_source = base_model_name
        else:
            tokenizer_source = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if is_adapter:
            base_model = AutoModelForCausalLM.from_pretrained(
                tokenizer_source,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.bfloat16,
                device_map=None,
                **load_kwargs,
            )
            adapter_state = None
            adapter_file = adapter_path / "adapter_model.safetensors"
            if adapter_file.exists():
                adapter_state = safe_load_file(str(adapter_file))
            else:
                adapter_file = adapter_path / "adapter_model.bin"
                if adapter_file.exists():
                    adapter_state = torch.load(adapter_file, map_location="cpu")
            if adapter_state:
                embed_key = next((k for k in adapter_state.keys() if k.endswith("embed_tokens.weight")), None)
                if embed_key:
                    adapter_vocab_size = adapter_state[embed_key].shape[0]
                    base_vocab_size = base_model.get_input_embeddings().num_embeddings
                    if adapter_vocab_size != base_vocab_size:
                        base_model.resize_token_embeddings(adapter_vocab_size)
            peft_model = PeftModel.from_pretrained(base_model, model_name)
            try:
                self.model = peft_model.merge_and_unload()
            except Exception as exc:
                # logger.warning("Failed to merge LoRA adapter, using PEFT wrapper directly: %s", exc)
                self.model = peft_model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.bfloat16,
                device_map=None,
                **load_kwargs,
            )
        self.model.to(self.device)
        self.model.eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        stop_token_ids = self.tokenizer.encode(stop_symbol, add_special_tokens=False)
        self.stop_token_id = stop_token_ids[-1] if len(stop_token_ids) > 0 else self.tokenizer.eos_token_id
        self.input_max_len = input_max_len
        self.default_chat_prefixes = {"assistant": "Assistant:", "user": "User:"}
        self.inference_args = {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.stop_token_id,
            # "no_repeat_ngram_size": 3,
        }

    def _prepare_generation_args(self, gen_args: Dict) -> Dict:
        gen_params = {**self.inference_args}
        gen_params.update(gen_args)
        legacy_max = gen_params.pop("max_tokens", None)
        if legacy_max is not None and gen_params.get("max_new_tokens") is None:
            gen_params["max_new_tokens"] = legacy_max
        legacy_num_ret = gen_params.pop("n", None)
        if legacy_num_ret is not None and gen_params.get("num_return_sequences") is None:
            gen_params["num_return_sequences"] = legacy_num_ret
        for legacy in ("return_fulLocalModell_text", "echo", "stop"):
            gen_params.pop(legacy, None)
        if gen_params.get("num_return_sequences", 1) < 1:
            gen_params["num_return_sequences"] = 1
        if gen_params.get("num_return_sequences", 1) > 1 and not gen_params.get("do_sample", False):
            gen_params["do_sample"] = True
        if gen_params.get("pad_token_id") is None:
            gen_params["pad_token_id"] = self.tokenizer.pad_token_id
        if gen_params.get("eos_token_id") is None:
            gen_params["eos_token_id"] = self.stop_token_id
        return gen_params

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        if not messages:
            return ""
        prompt_lines: List[str] = []
        assistant_prefix = None
        user_prefix = None
        for message in messages:
            content = message.get("content", "").strip()
            if not content:
                continue
            prompt_lines.append(content)
            colon_idx = content.find(":")
            if colon_idx != -1:
                prefix = content[:colon_idx + 1]
                if message.get("role") == "assistant":
                    assistant_prefix = prefix
                elif message.get("role") == "user":
                    user_prefix = prefix
        last_role = messages[-1].get("role")
        if last_role == "user":
            next_prefix = assistant_prefix or self.default_chat_prefixes["assistant"]
        elif last_role == "assistant":
            next_prefix = user_prefix or self.default_chat_prefixes["user"]
        else:
            next_prefix = assistant_prefix or self.default_chat_prefixes["assistant"]
        prompt_lines.append(f"{next_prefix} ")
        return "\n".join(prompt_lines)

    def generate(self, input_text: str, **gen_args):
        gen_params = self._prepare_generation_args(gen_args)
        inputs = self.tokenizer([input_text], return_tensors='pt', truncation=True, max_length=self.input_max_len)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs['input_ids'].shape[-1]

        with torch.no_grad():
            try:
                outputs = self.model.generate(**inputs, **gen_params)
            except ValueError as exc:
                msg = str(exc)
                if "not used by the model" in msg:
                    unused = re.findall(r"'([^']+)'", msg)
                    stripped_any = False
                    for key in unused:
                        if key in gen_params:
                            gen_params.pop(key, None)
                            stripped_any = True
                    if stripped_any:
                        outputs = self.model.generate(**inputs, **gen_params)
                    else:
                        raise
                else:
                    raise
        gen_only_outputs = outputs[:, prompt_len:].detach().cpu()
        gen_resps = self.tokenizer.batch_decode(gen_only_outputs, skip_special_tokens=True)
        gen_output = []
        for resp in gen_resps:
            gen_output.append({"generated_text": resp})
        return gen_output

    def _infer_num_actions(self) -> int:
        n_goals = getattr(self.model_config, "n_goals", 1)
        n_topics = getattr(self.model_config, "n_topics", 1)
        return n_goals * n_topics if self.model_config.combined_action else n_goals

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Encode context with the PLM, take the [CLS] token and project to action logits.
        """
        # Request hidden states so we can grab a stable representation from a causal LM.
        outputs = self.plm(**batch["context"], output_hidden_states=True, use_cache=False)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            cls_token = hidden_states[-1][:, 0, :]
        else:
            # Fallback: some implementations still expose last_hidden_state
            cls_token = getattr(outputs, "last_hidden_state", None)
            if cls_token is None:
                # As a last resort, derive a pseudo-embedding from logits
                cls_token = outputs.logits
                if cls_token.dim() == 3:
                    cls_token = cls_token[:, -1, :]
        cls_token = self.drop_out(cls_token)
        logits = self.out_layer(cls_token)
        return logits

    # Expose embedding accessors expected by TRL trainers.
    def get_input_embeddings(self):
        return self.plm.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.plm.set_input_embeddings(new_embeddings)

    # Expose gradient checkpointing toggles expected by HF trainers.
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.plm, "gradient_checkpointing_enable"):
            return self.plm.gradient_checkpointing_enable(**kwargs)
        return None

    def gradient_checkpointing_disable(self):
        if hasattr(self.plm, "gradient_checkpointing_disable"):
            return self.plm.gradient_checkpointing_disable()
        return None

    # Minimal generate wrapper so TRL DPOTrainer can call into the underlying LM.
    def generate(self, *args, **kwargs):
        return self.plm.generate(*args, **kwargs)
