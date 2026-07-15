from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast

from src.uav.model.connector import get_connector
from src.uav.model.motion_encoder import VideoFlowMotionEncoderConfig, VideoFlowMotionEncoderPreTrainedModel

# CLIP normalization constants (same as Qwen2.5-VL image preprocessing)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class MotionMLLMConfig(Qwen2_5_VLConfig):
    model_type = "motion-mllm"

    def __init__(self, motion_config=None, connector_config=None, **kwargs):
        super().__init__(**kwargs)
        # MOFNet: frozen, computes optical flow
        self.sub_configs["motion_config"] = VideoFlowMotionEncoderConfig
        if isinstance(motion_config, dict):
            self.motion_config = self.sub_configs["motion_config"](**motion_config)
        elif motion_config is None:
            self.motion_config = self.sub_configs["motion_config"]()

        self.connector_config = connector_config if connector_config is not None else {}


class MotionMLLMForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.motion_encoder = VideoFlowMotionEncoderPreTrainedModel(config.motion_config)
        self.connector = get_connector(config)

        # Initialize weights and apply final processing
        self.post_init()

    def _get_visual_module(self):
        visual = getattr(self, "visual", None)
        if visual is None and hasattr(self, "model"):
            visual = getattr(self.model, "visual", None)
        if visual is None:
            raise AttributeError("No visual module found on MotionMLLM/Qwen2.5-VL model")
        return visual

    def _encode_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "get_image_features"):
            image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
            if isinstance(image_embeds, (list, tuple)):
                image_embeds = torch.cat(image_embeds, dim=0)
            return image_embeds

        visual = self._get_visual_module()
        pixel_values = pixel_values.type(visual.dtype)
        return visual(pixel_values, grid_thw=image_grid_thw)

    def _encode_video_features(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "get_video_features"):
            video_embeds = self.model.get_video_features(pixel_values_videos, video_grid_thw)
            if isinstance(video_embeds, (list, tuple)):
                video_embeds = torch.cat(video_embeds, dim=0)
            return video_embeds

        visual = self._get_visual_module()
        pixel_values_videos = pixel_values_videos.type(visual.dtype)
        return visual(pixel_values_videos, grid_thw=video_grid_thw)

    def _scatter_multimodal_features(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        features: torch.Tensor,
        token_id: int,
    ) -> torch.Tensor:
        mask = input_ids == token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        feature_mask = mask_expanded.to(inputs_embeds.device)
        features = features.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(feature_mask, features)

    def _flow_to_visual_input(
        self,
        flow_list: List[torch.Tensor],
        video_grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert optical flow to pseudo-image input for Qwen ViT.

        Args:
            flow_list: List of [T, 4, H, W] optical flow tensors from MOFNet.
            video_grid_thw: [num_videos, 3] with (grid_t, grid_h, grid_w).

        Returns:
            pixel_values_flow: [total_patches, 1176] patchified pseudo-images.
                1176 = 3 channels × 2 temporal_patch × 14 × 14 spatial_patch
            flow_grid_thw: same as video_grid_thw (flow uses same grid layout).
        """
        temporal_patch_size = 2  # Qwen2.5-VL temporal patch size
        spatial_patch_size = 14  # Qwen2.5-VL spatial patch size

        # Register CLIP normalization tensors
        mean = torch.tensor(CLIP_MEAN, dtype=torch.float32)
        std = torch.tensor(CLIP_STD, dtype=torch.float32)

        all_patches = []
        grid_idx = 0

        for flow in flow_list:
            # flow: [T, 4, H, W] — take forward flow (first 2 channels)
            forward_flow = flow[:, :2]  # [T, 2, H, W]
            T, _, H, W = forward_flow.shape
            device = forward_flow.device

            # Construct 3 channels: (magnitude, dx, dy)
            dx = forward_flow[:, 0:1]  # [T, 1, H, W]
            dy = forward_flow[:, 1:2]  # [T, 1, H, W]
            magnitude = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)  # [T, 1, H, W]
            pseudo_img = torch.cat([magnitude, dx, dy], dim=1)  # [T, 3, H, W]

            # Per-video min-max normalization to [0, 1]
            vmin = pseudo_img.min()
            vmax = pseudo_img.max()
            if vmax - vmin > 1e-6:
                pseudo_img = (pseudo_img - vmin) / (vmax - vmin)
            else:
                pseudo_img = torch.zeros_like(pseudo_img)

            # Apply CLIP normalization
            mean_dev = mean.to(device).view(1, 3, 1, 1)
            std_dev = std.to(device).view(1, 3, 1, 1)
            pseudo_img = (pseudo_img - mean_dev) / std_dev

            # Get target resolution from grid_thw
            gt, gh, gw = video_grid_thw[grid_idx].tolist()
            gt, gh, gw = int(gt), int(gh), int(gw)
            grid_idx += 1

            target_H = gh * spatial_patch_size  # grid_h * 14
            target_W = gw * spatial_patch_size  # grid_w * 14
            target_T = gt * temporal_patch_size  # grid_t * 2

            # Resize spatial dimensions
            pseudo_img = F.interpolate(
                pseudo_img.float(), size=(target_H, target_W),
                mode='bilinear', align_corners=False,
            )  # [T, 3, target_H, target_W]

            # Temporal alignment to target_T
            if T < target_T:
                pad = pseudo_img[-1:].expand(target_T - T, -1, -1, -1)
                pseudo_img = torch.cat([pseudo_img, pad], dim=0)
            elif T > target_T:
                indices = torch.linspace(0, T - 1, target_T, dtype=torch.long, device=device)
                pseudo_img = pseudo_img[indices]
            # pseudo_img: [target_T, 3, target_H, target_W]

            # Patchify: match Qwen2.5-VL processor's merge-grouped order.
            # The ViT groups every spatial_merge_size^2 (=4) consecutive patches
            # into a merge block, so the order must be:
            # [gt, gh//merge, gw//merge, merge_h, merge_w, C, t_patch, H_patch, W_patch]
            spatial_merge_size = 2
            pseudo_img = pseudo_img.view(
                gt, temporal_patch_size,                                           # 0, 1
                3,                                                                 # 2
                gh // spatial_merge_size, spatial_merge_size, spatial_patch_size,   # 3, 4, 5
                gw // spatial_merge_size, spatial_merge_size, spatial_patch_size,   # 6, 7, 8
            )
            pseudo_img = pseudo_img.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
            # [gt, gh//2, gw//2, merge_h, merge_w, 3, 2, 14, 14]
            n_patches = gt * gh * gw
            pseudo_img = pseudo_img.view(n_patches, 3 * temporal_patch_size * spatial_patch_size * spatial_patch_size)
            # [n_patches, 1176]

            all_patches.append(pseudo_img)

        pixel_values_flow = torch.cat(all_patches, dim=0)
        flow_grid_thw = video_grid_thw.clone()

        return pixel_values_flow, flow_grid_thw

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_tchw: Optional[List[torch.FloatTensor]] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

        # If no motion features, delegate entirely to parent
        if video_tchw is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                cache_position=cache_position,
                second_per_grid_ts=second_per_grid_ts,
                **kwargs,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

            # Handle images
            if pixel_values is not None:
                image_embeds = self._encode_image_features(pixel_values, image_grid_thw)
                inputs_embeds = self._scatter_multimodal_features(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    features=image_embeds,
                    token_id=self.config.image_token_id,
                )

            # Handle videos with motion fusion
            if pixel_values_videos is not None:
                # Visual encoder is typically frozen; skip graph to save memory.
                visual = self._get_visual_module()
                if not hasattr(self, "_visual_requires_grad"):
                    self._visual_requires_grad = any(p.requires_grad for p in visual.parameters())
                if self._visual_requires_grad:
                    video_embeds = self._encode_video_features(pixel_values_videos, video_grid_thw)
                else:
                    with torch.no_grad():
                        video_embeds = self._encode_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

                # Motion pipeline: MOFNet (frozen) → optical flow → ViT (frozen) → flow_embeds → Connector (trainable)
                me_device = next(self.motion_encoder.parameters()).device
                video_tchw = [v.to(me_device).contiguous() for v in video_tchw]
                # Most videos run faster with cuDNN enabled. Only fall back to
                # the slower non-cuDNN kernels when a sample explicitly asks
                # for it, or when cuDNN rejects the current tensor layout.
                disable_motion_cudnn = getattr(self, "_disable_motion_cudnn_fallback", False)
                try:
                    if disable_motion_cudnn:
                        with torch.backends.cudnn.flags(enabled=False):
                            flow_list = self.motion_encoder(video_tchw)
                    else:
                        flow_list = self.motion_encoder(video_tchw)
                except RuntimeError as exc:
                    if "CUDNN_STATUS_NOT_SUPPORTED" not in str(exc):
                        raise
                    with torch.backends.cudnn.flags(enabled=False):
                        flow_list = self.motion_encoder(video_tchw)

                # Sanitize optical flow: MOFNet can produce NaN/Inf for
                # corrupted frames or extreme motion. Clamp to bf16-safe range.
                flow_list = [torch.nan_to_num(f.clamp(-400, 400), nan=0.0, posinf=400.0, neginf=-400.0)
                             for f in flow_list]

                # Convert flow to ViT pseudo-image input and get flow embeddings (frozen)
                with torch.no_grad():
                    pixel_values_flow, flow_grid_thw = self._flow_to_visual_input(flow_list, video_grid_thw)
                    # Move to ViT device
                    vit_device = next(visual.parameters()).device
                    pixel_values_flow = pixel_values_flow.to(vit_device, dtype=pixel_values_videos.dtype)
                    flow_grid_thw = flow_grid_thw.to(vit_device)
                    flow_embeds = self._encode_video_features(pixel_values_flow, flow_grid_thw)
                flow_embeds = flow_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

                # Forward diagnostic: check ViT outputs for NaN
                if self.training:
                    for _name, _t in [("flow_embeds", flow_embeds), ("video_embeds", video_embeds)]:
                        if not torch.isfinite(_t).all():
                            _n = (~torch.isfinite(_t)).sum().item()
                            print(f"[FWD CHECK] {_name}: {_n}/{_t.numel()} NaN/Inf")

                # Safety: ViT can produce NaN for OOD flow pseudo-images in bf16
                flow_embeds = torch.nan_to_num(flow_embeds, nan=0.0, posinf=0.0, neginf=0.0)

                # Connector fusion (trainable)
                # Account for ViT spatial merger (2×2 patches → 1 token):
                # post-merger tokens = gt * (gh // merge) * (gw // merge)
                spatial_merge_size = visual.spatial_merge_size
                merged_grid_thw = video_grid_thw.clone()
                merged_grid_thw[:, 1] = merged_grid_thw[:, 1] // spatial_merge_size
                merged_grid_thw[:, 2] = merged_grid_thw[:, 2] // spatial_merge_size

                c_device = next(self.connector.parameters()).device
                fused_embeds = self.connector(
                    visual_embeds=video_embeds.to(c_device),
                    flow_embeds=flow_embeds.to(c_device),
                    grid_thw=merged_grid_thw.to(c_device),
                )

                # Align back to inputs_embeds device/dtype
                fused_embeds = fused_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

                # Safety: if fused_embeds has NaN, fall back to visual_embeds
                # (preserves visual info instead of replacing with 0)
                if not torch.isfinite(fused_embeds).all():
                    if self.training:
                        _n = (~torch.isfinite(fused_embeds)).sum().item()
                        print(f"[FWD CHECK] fused_embeds: {_n}/{fused_embeds.numel()} NaN/Inf → fallback to visual_embeds")
                    _finite_mask = torch.isfinite(fused_embeds)
                    fused_embeds = torch.where(_finite_mask, fused_embeds, video_embeds.to(fused_embeds.device))

                # Gradient hook: clamp NaN/Inf from LLM backward to prevent
                # poisoning connector gradients (return clean gradient)
                if fused_embeds.requires_grad and self.training:
                    def _clamp_fused_grad(grad):
                        if not torch.isfinite(grad).all():
                            n_bad = (~torch.isfinite(grad)).sum().item()
                            print(f"[GRAD CLAMP] fused_embeds: {n_bad}/{grad.numel()} NaN/Inf → zeroed")
                            return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                        return grad
                    fused_embeds.register_hook(_clamp_fused_grad)

                inputs_embeds = self._scatter_multimodal_features(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    features=fused_embeds,
                    token_id=self.config.video_token_id,
                )

        # Delegate the remaining language-model forward to the parent class.
        # This keeps us compatible with multiple transformers variants where
        # Qwen2.5-VL internals differ (`self.visual` vs `self.model.visual`,
        # presence/absence of multimodal helpers, different text-model forward signature).
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        video_tchw=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # Pass video_tchw only on prefill step; clear on decode steps
        if cache_position is not None and cache_position[0] != 0:
            model_inputs["video_tchw"] = None
        else:
            model_inputs["video_tchw"] = video_tchw

        return model_inputs
