"""Ideogram 4 (Qwen3-VL-8B, 13-layer tap) specific conditioning rebalance nodes."""

import torch

from . import conditioning_rebalance as core
from .conditioning_rebalance import (
    compile_edit,
    guidance,
    refocus,
    merge_conditioning_anchor,
    merge_conditioning_multi,
    _align_prompt,
)

try:
    import comfy.utils
    import node_helpers
    _COMFY_AVAILABLE = True
except ImportError:
    _COMFY_AVAILABLE = False


# 13-layer tap of Qwen3-VL-8B (comfy captures layer inputs, offset by +1).
IDEOGRAM4_TAP_LAYERS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 36]
IDEOGRAM4_N_TAPS = len(IDEOGRAM4_TAP_LAYERS)            # 13
IDEOGRAM4_HIDDEN_DIM = 4096
IDEOGRAM4_FEATURE_DIM = IDEOGRAM4_N_TAPS * IDEOGRAM4_HIDDEN_DIM   # 53248

# Register the Ideogram 4 profile with the core detection system.
core.register_encoder_profile(
    "ideogram4",
    n_taps=IDEOGRAM4_N_TAPS,
    hidden_dim=IDEOGRAM4_HIDDEN_DIM,
    tap_layers=IDEOGRAM4_TAP_LAYERS,
)

# Ignored
IDEOGRAM4_SYS_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def compile_edit_ideogram4(clip, prompt, images_with_size=None):

    return compile_edit(clip, prompt, images_with_size, llama_template=None)


class ConditioningIdeogram4Rebalance:

    # 13 weights
    DEFAULT_WEIGHTS = "1.0,1.0,1.0,1.0,1.0,0.0,2.25,0.0,2.25,0.5,1.0,1.0,1.0"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "multiplier": ("FLOAT", {"default": 4.0, "min": -1000000000.0, "max": 1000000000.0, "step": 0.01}),
            "per_layer_weights": ("STRING", {"default": cls.DEFAULT_WEIGHTS, "multiline": False}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/conditioning"

    def main(self, conditioning, multiplier, per_layer_weights=None):
        plw = core._parse_floats(per_layer_weights) if per_layer_weights else None
        c = core.scale_conditioning(conditioning, multiplier, weights=plw)
        return (c,)


class Ideogram4EditRebalance:

    # 13 weights
    DEFAULT_MAIN_WEIGHTS = "1.0,1.0,1.0,1.0,1.0,0.0,2.25,0.0,2.25,0.5,1.0,1.0,1.0"
    DEFAULT_REF_WEIGHTS = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP",),
            "steering": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            "layer_multiplier": ("FLOAT", {"default": 1.0, "min": -1000000000.0, "max": 1000000000.0, "step": 0.01}),
            "enable_step": ("BOOLEAN", {"default": True}),
        }, "optional": {
            "image1": ("IMAGE",),
            "image1_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image2": ("IMAGE",),
            "image2_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image3": ("IMAGE",),
            "image3_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image4": ("IMAGE",),
            "image4_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    OUTPUT_IS_LIST = (True,)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/conditioning"

    @staticmethod
    def _batch_len(image):
        """Return the batch length of an image tensor or list (0 if None)."""
        if image is None:
            return 0
        if isinstance(image, list):
            return len(image)
        if hasattr(image, "shape") and len(image.shape) >= 4:
            return int(image.shape[0])
        if hasattr(image, "shape") and len(image.shape) == 3:
            return 1
        return 1

    @staticmethod
    def _slice_image(image, idx):
        """Return the idx-th frame of an image tensor/list, keeping a batch dim."""
        if image is None:
            return None
        if isinstance(image, list):
            item = image[idx]
            if hasattr(item, "shape") and len(item.shape) == 3:
                return item.unsqueeze(0)
            return item
        if len(image.shape) >= 4:
            return image[idx:idx + 1]
        return image

    @staticmethod
    def _image_signature(image):
        """Signature for caching: shape + a hash of the pixels."""
        return core.image_signature(image)

    @staticmethod
    def _list_index():
        """Comfy list-map index for this invocation, or None."""
        try:
            from comfy_execution.utils import get_executing_context
            ctx = get_executing_context()
            if ctx is None:
                return None
            return ctx.list_index
        except Exception:
            return None

    def main(self, text, clip,
             steering=1.0,
             layer_multiplier=1.0,
             enable_step=True,
             negative=None, anchor=None,
             image1=None, image1_tokens="normal",
             image2=None, image2_tokens="normal",
             image3=None, image3_tokens="normal",
             image4=None, image4_tokens="normal"):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Ideogram 4 Edit requires ComfyUI (comfy.utils, node_helpers).")


        list_index = self._list_index()
        if list_index is not None and list_index > 0:
            return ([],)

        match_percent = 0.8

        prompt = "" + _align_prompt(text)
        ref_prefix = negative if negative is not None and str(negative) != "" else ""
        prompt_ref = str(ref_prefix) + ""

        ml = self.DEFAULT_MAIN_WEIGHTS
        rl = self.DEFAULT_REF_WEIGHTS

        image_slots = [
            (image1, image1_tokens),
            (image2, image2_tokens),
            (image3, image3_tokens),
            (image4, image4_tokens),
        ]

        max_batch = 0
        for img, _ in image_slots:
            max_batch = max(max_batch, self._batch_len(img))

        n_passes = max_batch if max_batch > 0 else 1

        guidance_passes = []

        # Bounded LRU + CLIP identity in the key (see Krea2EditRebalance).
        cache = getattr(self, "_pass_cache", None)
        if not isinstance(cache, core.PassCache):
            cache = core.PassCache()
            self._pass_cache = cache
        clip_sig = core.clip_signature(clip)

        for p in range(n_passes):
            pass_images = []
            for img, tier in image_slots:
                if img is None:
                    pass_images.append((None, tier))
                    continue
                bl = self._batch_len(img)
                if bl > 1:
                    pass_images.append((self._slice_image(img, p), tier))
                else:
                    pass_images.append((img, tier))

            has_image = any(img is not None for img, _ in pass_images)

            key = (
                clip_sig,
                p,
                tuple(self._image_signature(img) for img, _ in pass_images),
                tuple(tier for _, tier in pass_images),
                float(steering),
                float(layer_multiplier), str(ml), str(rl),
                bool(enable_step),
                prompt, prompt_ref,
            )

            cached = cache.get(key)
            if cached is not None:
                guidance_passes.append(cached)
                continue

            cond_main = compile_edit_ideogram4(clip, prompt, pass_images if has_image else None)
            cond_ref = compile_edit_ideogram4(clip, prompt_ref, pass_images if has_image else None)

            cond_main = refocus(cond_main, layer_multiplier, ml)
            cond_ref = refocus(cond_ref, layer_multiplier, rl)

            pass_guidance = guidance(cond_main, cond_ref, steering)
            cache.put(key, pass_guidance)
            guidance_passes.append(pass_guidance)

        if not guidance_passes:
            final = compile_edit_ideogram4(clip, prompt, None)
            return ([final],)

        if len(guidance_passes) == 1:
            merged = guidance_passes[0]
        elif anchor is not None:
            merged = merge_conditioning_anchor(anchor, guidance_passes, match_percent)
        else:
            merged = merge_conditioning_multi(guidance_passes, match_percent)

        if enable_step:
            cond_raw = compile_edit_ideogram4(clip, prompt, None)
            merged = core.RebalanceCFG().main(
                cond_raw, merged,
                "0.000-0.100:4.00;",
                "0.100-0.750:0.80; 0.750-0.875:1.40; 0.875-1.000:20.50",
                "gradual", 8,
            )[0]

        return ([merged],)


class Ideogram4EncodeRebalance:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP",),
        },
        "optional": {
            "image1": ("IMAGE",),
            "image1_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image2": ("IMAGE",),
            "image2_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image3": ("IMAGE",),
            "image3_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image4": ("IMAGE",),
            "image4_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/conditioning"

    def main(self, text, clip,
             image1=None, image1_tokens="normal",
             image2=None, image2_tokens="normal",
             image3=None, image3_tokens="normal",
             image4=None, image4_tokens="normal"):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Ideogram 4 Edit requires ComfyUI (comfy.utils, node_helpers).")

        prompt = "" + _align_prompt(text)

        images_with_size = [
            (image1, image1_tokens),
            (image2, image2_tokens),
            (image3, image3_tokens),
            (image4, image4_tokens),
        ]
        has_image = any(img is not None for img, _ in images_with_size)

        final = compile_edit_ideogram4(clip, prompt, images_with_size if has_image else None)

        return (final,)


NODE_CLASS_MAPPINGS = {
    "ConditioningIdeogram4Rebalance": ConditioningIdeogram4Rebalance,
    "Ideogram4EditRebalance": Ideogram4EditRebalance,
    "Ideogram4EncodeRebalance": Ideogram4EncodeRebalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningIdeogram4Rebalance": "Conditioning Ideogram4 Rebalance",
    "Ideogram4EditRebalance": "Ideogram 4 Image Edit Rebalance",
    "Ideogram4EncodeRebalance": "Ideogram 4 Encode Rebalance",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "ConditioningIdeogram4Rebalance",
    "Ideogram4EditRebalance",
    "Ideogram4EncodeRebalance",
    "IDEOGRAM4_TAP_LAYERS",
    "IDEOGRAM4_FEATURE_DIM",
]
