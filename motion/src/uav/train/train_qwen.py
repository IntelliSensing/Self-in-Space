# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import logging
import os
import pathlib
import shutil
import sys
from pathlib import Path

# add repo root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
)

import src.uav.train.trainer
from src.uav.data.data_processor import make_supervised_data_module
from src.uav.model.motion_mllm import MotionMLLMConfig, MotionMLLMForConditionalGeneration
from src.uav.train.argument import DataArguments, ModelArguments, TrainingArguments


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _unwrap_model(model):
    """Unwrap DeepSpeed / PEFT wrappers to get the base MotionMLLM model."""
    # DeepSpeed wraps model in engine.module
    if hasattr(model, "module"):
        model = model.module
    # PEFT wraps model in base_model.model (PeftModel → LoraModel → original)
    # Note: PreTrainedModel also has a `base_model` property, so we must check
    # for PeftModel specifically to avoid false positives in non-LoRA stages.
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            model = model.base_model.model
    except ImportError:
        pass
    return model


class ConnectorMonitorCallback(TrainerCallback):
    """Log connector metrics and save connector weights at each checkpoint.

    Metrics logged:
      - connector/mlp_weight_norm: MLP total weight norm
      - connector/mlp_grad_norm: MLP total gradient norm (captured via backward hook)
      - connector/cross_attn_weight_norm: Cross-attention total weight norm
      - connector/cross_attn_grad_norm: Cross-attention total gradient norm
    """

    def __init__(self):
        super().__init__()
        self._last_mlp_grad_norm = 0.0
        self._last_cross_attn_grad_norm = 0.0
        self._hooks = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Register backward hooks to capture metrics."""
        if model is None:
            return
        base = _unwrap_model(model)
        if not hasattr(base, "connector"):
            print("[ConnectorMonitor] WARNING: no connector found on unwrapped model!")
            return

        connector = base.connector

        # Verify connector params are finite after all wrapping (PEFT + DeepSpeed)
        for name, p in connector.named_parameters():
            if not torch.isfinite(p.data).all():
                print(f"[ConnectorMonitor] WARNING: {name} contains NaN/Inf! shape={p.shape}, dtype={p.dtype}")
            else:
                print(f"[ConnectorMonitor] OK: {name} finite, norm={p.data.float().norm().item():.6f}")

        # --- Gradient sanitization hooks (MUST be registered FIRST) ---
        # With DeepSpeed bf16, gradient overflow (>65504) in ANY parameter can
        # produce NaN that contaminates ALL gradients via gradient clipping
        # (scale = 1/grad_norm, if grad_norm includes NaN → scale=NaN → all grads NaN).
        # This hook replaces NaN/Inf gradients with 0 before they reach the optimizer.
        for name, p in connector.named_parameters():
            if p.requires_grad:
                def _sanitize_grad(grad, param_name=name):
                    if not torch.isfinite(grad).all():
                        n_bad = (~torch.isfinite(grad)).sum().item()
                        print(f"[GRAD SANITIZE] {param_name}: {n_bad}/{grad.numel()} NaN/Inf values")
                        return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                self._hooks.append(p.register_hook(_sanitize_grad))

        # --- Monitoring hooks (after sanitization) ---
        # MLP gradient hook (check flow_mlp for VisualFlowConnector, or mlp for legacy)
        mlp_module = getattr(connector, "flow_mlp", None) or getattr(connector, "mlp", None)
        if mlp_module is not None:
            mlp_params = [p for p in mlp_module.parameters() if p.requires_grad]
            if mlp_params:
                def _mlp_hook(grad, self_ref=self, params=mlp_params):
                    total_sq = 0.0
                    for p in params:
                        if p.grad is not None:
                            total_sq += p.grad.data.float().norm().item() ** 2
                    total_sq += grad.float().norm().item() ** 2
                    self_ref._last_mlp_grad_norm = total_sq ** 0.5
                self._hooks.append(mlp_params[-1].register_hook(_mlp_hook))

        # Cross-attention gradient hook
        if hasattr(connector, "cross_attn"):
            ca_params = [p for p in connector.cross_attn.parameters() if p.requires_grad]
            if ca_params:
                def _ca_hook(grad, self_ref=self, params=ca_params):
                    total_sq = 0.0
                    for p in params:
                        if p.grad is not None:
                            total_sq += p.grad.data.float().norm().item() ** 2
                    total_sq += grad.float().norm().item() ** 2
                    self_ref._last_cross_attn_grad_norm = total_sq ** 0.5
                self._hooks.append(ca_params[-1].register_hook(_ca_hook))

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None or logs is None:
            return
        base = _unwrap_model(model)
        if not hasattr(base, "connector"):
            return

        connector = base.connector
        connector_logs = {}

        # MLP weight norm and gradient norm (flow_mlp for VisualFlowConnector, mlp for legacy)
        mlp_module = getattr(connector, "flow_mlp", None) or getattr(connector, "mlp", None)
        if mlp_module is not None:
            mlp_sq = sum(p.data.float().norm().item() ** 2 for p in mlp_module.parameters())
            connector_logs["connector/mlp_weight_norm"] = mlp_sq ** 0.5
            connector_logs["connector/mlp_grad_norm"] = self._last_mlp_grad_norm

        # Cross-attention weight norm and gradient norm
        if hasattr(connector, "cross_attn"):
            ca_sq = sum(p.data.float().norm().item() ** 2 for p in connector.cross_attn.parameters())
            connector_logs["connector/cross_attn_weight_norm"] = ca_sq ** 0.5
            connector_logs["connector/cross_attn_grad_norm"] = self._last_cross_attn_grad_norm

        # Add connector metrics to Trainer logs and trainer_state.json.
        logs.update(connector_logs)
        # Also patch the already-appended log_history entry for trainer_state.json
        if state.log_history:
            state.log_history[-1].update(connector_logs)

    def on_save(self, args, state, control, model=None, **kwargs):
        """Save connector weights alongside each checkpoint."""
        if model is None or not args.should_save:
            return
        base = _unwrap_model(model)
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if hasattr(base, "connector"):
            connector_state = {k: v.cpu() for k, v in base.connector.state_dict().items()}
            torch.save(connector_state, os.path.join(ckpt_dir, "connector_weights.pt"))
            print(f"Connector weights saved to checkpoint-{state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        """Clean up backward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def build_callbacks(training_args):
    callbacks = [ConnectorMonitorCallback()]
    if not env_flag("SWANLAB_ENABLED", default=False):
        print("SwanLab logging disabled (set SWANLAB_ENABLED=1 to enable).")
        return callbacks

    swanlab_mode = os.getenv("SWANLAB_MODE", "cloud")
    try:
        from swanlab.integration.transformers import SwanLabCallback
    except Exception as exc:
        raise RuntimeError(
            "SWANLAB_ENABLED=1 but SwanLab is unavailable. "
            "Install/configure SwanLab or set SWANLAB_ENABLED=0."
        ) from exc

    try:
        callbacks.append(
            SwanLabCallback(
                project=os.getenv("SWANLAB_PROJECT", "motion-mllm"),
                experiment_name=os.path.basename(training_args.output_dir),
                mode=swanlab_mode,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize SwanLab callback. "
            "Check SwanLab credentials/network or set SWANLAB_ENABLED=0."
        ) from exc

    print(f"SwanLab logging enabled (mode={swanlab_mode}).")
    return callbacks


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    # visual.merger 始终冻结：merger 在 fusion 之前已完成投影，
    # 融合后的 token 直接输入 LLM，不再过 merger（与 Spatial-MLLM 一致）
    for n, p in model.visual.merger.named_parameters():
        p.requires_grad = False

    # New transformers: model.model is Qwen2_5_VLModel (visual + language_model)
    # LLM params are under model.model.language_model
    llm = getattr(model.model, "language_model", model.model)
    if model_args.tune_mm_llm:
        for n, p in llm.named_parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
    else:
        for n, p in llm.named_parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False

    if hasattr(model, "motion_encoder"):
        # MOFNet always frozen — only used for optical flow computation
        for n, p in model.motion_encoder.named_parameters():
            p.requires_grad = False

    if hasattr(model, "connector"):
        if model_args.tune_mm_connector:
            for n, p in model.connector.named_parameters():
                p.requires_grad = True
        else:
            for n, p in model.connector.named_parameters():
                p.requires_grad = False


def _reinit_new_modules(model):
    """Re-initialize connector after from_pretrained.

    HuggingFace's from_pretrained uses a _no_init_weights context manager
    that turns all nn.init.* calls into no-ops.  New modules (not in the
    pretrained checkpoint) end up with uninitialized / NaN parameters.
    This function explicitly re-initializes them.
    """
    # --- Connector (plain nn.Module — generic re-init) ---
    c = model.connector
    for module in c.modules():
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            nn.init.trunc_normal_(module.out_proj.weight, std=0.02)
            nn.init.zeros_(module.out_proj.bias)

    # Keep the new flow-attention branch close to zero at initialization so
    # training starts from the visual-only baseline instead of a noisy residual.
    # Only zero the final attention projection; zeroing the flow MLP output layer
    # blocks gradients from reaching the MLP at startup.
    cross_attn = getattr(c, "cross_attn", None)
    if isinstance(cross_attn, nn.MultiheadAttention):
        nn.init.zeros_(cross_attn.out_proj.weight)
        if cross_attn.out_proj.bias is not None:
            nn.init.zeros_(cross_attn.out_proj.bias)

    # Verify initialization produced valid (non-NaN) values
    c_ok = all(torch.isfinite(p).all() for p in model.connector.parameters())
    c_norm = sum(p.data.norm().item() ** 2 for p in model.connector.parameters()) ** 0.5
    print(f"[REINIT] connector: finite={c_ok}, norm={c_norm:.4f}")
    if not c_ok:
        raise RuntimeError("Re-initialization failed: parameters contain NaN/Inf!")


def get_model(model_args, data_args, training_args, attn_implementation="flash_attention_2"):
    if model_args.model_type.lower() != "motion-mllm":
        raise ValueError("The Motion training entry only supports model_type='motion-mllm'.")

    motion_mllm_config = MotionMLLMConfig.from_pretrained(
        model_args.pretrained_model_name_or_path,
        motion_config={
            "motion_dim": 128,
            "down_ratio": 8,
            "decoder_depth": 12,
        },
        connector_config={
            "connector_type": model_args.connector_type,
        },
    )
    model = MotionMLLMForConditionalGeneration.from_pretrained(
        model_args.pretrained_model_name_or_path,
        config=motion_mllm_config,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    model.motion_encoder.load_pretrained_weights(model_args.videoflow_checkpoints_path)
    device = next(model.parameters()).device
    model.motion_encoder.to(device=device, dtype=torch.float32)
    model.connector.to(device=device, dtype=torch.float32)
    _reinit_new_modules(model)

    processor = AutoProcessor.from_pretrained(
        model_args.pretrained_model_name_or_path,
    )
    return model, processor


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    model, processor = get_model(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        attn_implementation=attn_implementation,
    )

    data_args.model_type = "qwen2.5vl"

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.pretrained_model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        # Freeze all parameters first
        for p in model.parameters():
            p.requires_grad = False

        # Apply LoRA to LLM attention layers
        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        # Manually unfreeze trainable modules (PEFT's modules_to_save
        # wraps them with AuxiliaryTrainingWrapper which breaks multi-arg forward)
        base = model.base_model.model
        if hasattr(base, "connector"):
            for p in base.connector.parameters():
                p.requires_grad = True
            print("Connector parameters unfrozen manually.")
    else:
        set_model(model_args, model)

    # print trainable parameters
    if torch.distributed.get_rank() == 0:
        base_model = _unwrap_model(model)
        # visual lives under base_model.model (Qwen2_5_VLModel)
        if hasattr(base_model, "model") and hasattr(base_model.model, "visual"):
            base_model.model.visual.print_trainable_parameters()
            base_model.model.print_trainable_parameters()
        if hasattr(base_model, "motion_encoder"):
            base_model.motion_encoder.print_trainable_parameters()
        if hasattr(base_model, "connector"):
            base_model.connector.print_trainable_parameters()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        if training_args.lora_enable:
            model.print_trainable_parameters()

    data_module = make_supervised_data_module(processor=processor, data_args=data_args)

    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args,
        callbacks=build_callbacks(training_args),
        **data_module,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_state()

    # Copy chat_template.json to output dir (resolve HuggingFace cache path if needed)
    try:
        from huggingface_hub import hf_hub_download
        source_path = os.path.join(model_args.pretrained_model_name_or_path, "chat_template.json")
        if not os.path.exists(source_path):
            source_path = hf_hub_download(model_args.pretrained_model_name_or_path, "chat_template.json")
        template_path = os.path.join(training_args.output_dir, "chat_template.json")
        shutil.copy2(source_path, template_path)
    except Exception as e:
        print(f"Warning: could not copy chat_template.json: {e}")

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # Manually save connector weights (not included in LoRA adapter)
    if training_args.should_save:
        base = _unwrap_model(model)
        if hasattr(base, "connector"):
            connector_state = {k: v.cpu() for k, v in base.connector.state_dict().items()}
            connector_path = os.path.join(training_args.output_dir, "connector_weights.pt")
            torch.save(connector_state, connector_path)
            print(f"Connector weights saved to {connector_path}")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
