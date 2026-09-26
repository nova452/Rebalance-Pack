"""Krea 2 (Qwen3-VL-4B, 12-layer tap) specific conditioning rebalance nodes."""

import torch

from . import conditioning_rebalance as core
from .conditioning_rebalance import (
    compile_edit,
    guidance,
    refocus,
    merge_conditioning_multi,
    _align_prompt,
)

try:
    import comfy.utils
    import node_helpers
    _COMFY_AVAILABLE = True
except ImportError:
    _COMFY_AVAILABLE = False


# 12-layer tap of Qwen3-VL-4B (tap k == hidden_states[k], no offset).
KREA2_TAP_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
KREA2_N_TAPS = len(KREA2_TAP_LAYERS)          # 12
KREA2_HIDDEN_DIM = 2560
KREA2_FEATURE_DIM = KREA2_N_TAPS * KREA2_HIDDEN_DIM   # 30720

# Register the Krea 2 profile with the core detection system.
core.register_encoder_profile(
    "krea2",
    n_taps=KREA2_N_TAPS,
    hidden_dim=KREA2_HIDDEN_DIM,
    tap_layers=KREA2_TAP_LAYERS,
)

# System template used by Krea 2 image-edit conditioning.
KREA2_SYS_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def compile_edit_krea2(clip, prompt, images_with_size=None, fixed_res=None):
    """Encode a Krea 2 edit prompt with optional reference images.

    ``fixed_res`` overrides the per-tier resolution so every image is scaled to
    a fixed longest-side resolution (e.g. 32px for the low-res base encoding).
    """
    return compile_edit(clip, prompt, images_with_size, llama_template=KREA2_SYS_TEMPLATE, fixed_res=fixed_res)


def _build_steering_schedule(cond_main, cond_ref, points, interpolation="gradual", sub_steps=8):
    """Build time-scheduled steering segments from a parsed schedule.

    Each schedule point ``(start, end, strength)`` applies ``guidance`` at that
    steering strength, then the segment is tagged with ``start_percent`` /
    ``end_percent`` so the sampler honours it over the matching timestep range.
    ``gradual`` interpolation ramps the steering strength between adjacent
    points using ``sub_steps`` sub-segments (mirroring RebalanceCFG).
    """
    if not _COMFY_AVAILABLE:
        raise RuntimeError("Krea 2 steering schedule requires ComfyUI (node_helpers).")

    out = []
    n = len(points)
    for i, (t0, t1, m_i) in enumerate(points):
        m_next = points[i + 1][2] if i + 1 < n else m_i
        if t1 <= t0:
            t1 = max(t1, t0)
            seg = guidance(cond_main, cond_ref, m_i)
            seg = node_helpers.conditioning_set_values(
                seg, {"start_percent": t0, "end_percent": t1},
            )
            out.append(seg)
            continue

        if interpolation == "gradual" and sub_steps > 1 and m_next != m_i:
            for k in range(sub_steps):
                f0 = k / sub_steps
                f1 = (k + 1) / sub_steps
                ts = t0 + (t1 - t0) * f0
                te = t0 + (t1 - t0) * f1
                # use the sub-segment midpoint for a stable linear ramp
                m = m_i + (m_next - m_i) * ((f0 + f1) / 2.0)
                seg = guidance(cond_main, cond_ref, m)
                seg = node_helpers.conditioning_set_values(
                    seg, {"start_percent": ts, "end_percent": te},
                )
                out.append(seg)
        else:
            seg = guidance(cond_main, cond_ref, m_i)
            seg = node_helpers.conditioning_set_values(
                seg, {"start_percent": t0, "end_percent": t1},
            )
            out.append(seg)

    combined = []
    for seg in out:
        combined = combined + seg
    return combined


class ConditioningKrea2Rebalance:
    """Per-layer conditioning scaler for Krea 2's layout."""

    DEFAULT_WEIGHTS = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0"

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


class Krea2EncodeRebalance:

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
            raise RuntimeError("Krea 2 Edit requires ComfyUI (comfy.utils, node_helpers).")

        prompt = "" + _align_prompt(text)

        images_with_size = [
            (image1, image1_tokens),
            (image2, image2_tokens),
            (image3, image3_tokens),
            (image4, image4_tokens),
        ]
        has_image = any(img is not None for img, _ in images_with_size)

        final = compile_edit_krea2(clip, prompt, images_with_size if has_image else None)

        return (final,)


class Krea2EditRebalance:

    DEFAULT_MAIN_WEIGHTS = "0,0,0,0,0,0,0,0,12,0,0,0"
    DEFAULT_REF_WEIGHTS = "1.0909,1.0909,1.0909,1.0909,1.0909,1.0909,1.0909,1.0909,0,1.0909,1.0909,1.0909"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP",),
            "steering": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
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
        # List of image tensors (e.g. from LoadImages output list)
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
            # Ensure a batch dimension
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

        try:
            from comfy_execution.utils import get_executing_context
            ctx = get_executing_context()
            if ctx is None:
                return None
            return ctx.list_index
        except Exception:
            return None

    def main(self, text, clip,
             steering=0.0,
             layer_multiplier=1.0,
             enable_step=True,
             negative=None,
             image1=None, image1_tokens="normal",
             image2=None, image2_tokens="normal",
             image3=None, image3_tokens="normal",
             image4=None, image4_tokens="normal"):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Krea 2 Encode requires ComfyUI (comfy.utils, node_helpers).")

        list_index = self._list_index()
        if list_index is not None and list_index > 0:
            return ([],)

        match_percent = 0.8

        prompt = "" + _align_prompt(text)
        ref_prefix = negative if negative is not None and str(negative) != "" else ""
        prompt_ref = str(ref_prefix) + ""

        image_slots = [
            (image1, image1_tokens),
            (image2, image2_tokens),
            (image3, image3_tokens),
            (image4, image4_tokens),
        ]

        # Determine the pass count from the largest image batch.
        max_batch = 0
        for img, _ in image_slots:
            max_batch = max(max_batch, self._batch_len(img))

        n_passes = max_batch if max_batch > 0 else 1

        guidance_passes = []

        # Per-pass cache: keyed on a signature of the pass's images + params.
        # Minor option tweaks reuse cached passes instead of re-encoding.
        # Bounded LRU (old unbounded dict leaked GPU memory forever) and the
        # key includes the CLIP identity so model swaps invalidate it.
        cache = getattr(self, "_pass_cache", None)
        if not isinstance(cache, core.PassCache):
            cache = core.PassCache()
            self._pass_cache = cache
        clip_sig = core.clip_signature(clip)

        for p in range(n_passes):
            # Build per-pass image list: slice any batched input to frame p,
            # keep single/non-batched inputs as-is.
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

            # Cache key: image signatures + the encode/rebalance params.
            key = (
                clip_sig,
                p,
                tuple(self._image_signature(img) for img, _ in pass_images),
                tuple(tier for _, tier in pass_images),
                float(steering),
                float(layer_multiplier),
                bool(enable_step),
                prompt, prompt_ref,
            )

            cached = cache.get(key)
            if cached is not None:
                guidance_passes.append(cached)
                continue

            cond_main = compile_edit_krea2(clip, prompt, pass_images if has_image else None)
            cond_ref = compile_edit_krea2(clip, prompt_ref, pass_images if has_image else None)

            # Refocus main and ref with the shared multiplier + fixed layers.
            cond_main = refocus(cond_main, layer_multiplier, self.DEFAULT_MAIN_WEIGHTS)
            cond_ref = refocus(cond_ref, layer_multiplier, self.DEFAULT_REF_WEIGHTS)

            # First guidance for this pass.
            pass_guidance = guidance(cond_main, cond_ref, steering)
            cache.put(key, pass_guidance)
            guidance_passes.append(pass_guidance)

        if not guidance_passes:
            # Fallback: encode text-only.
            final = compile_edit_krea2(clip, prompt, None)
            return ([final],)

        if len(guidance_passes) == 1:
            merged = guidance_passes[0]
        else:
            # Always fall back to Conditioning Merge (Multi).
            merged = merge_conditioning_multi(guidance_passes, match_percent)

        if enable_step:
            # Custom Rebalance CFG with fixed schedules.
            cond_raw = compile_edit_krea2(clip, prompt, None)
            merged = core.RebalanceCFG().main(
                cond_raw, merged,
                "0.000-0.125:4.00;",
                "0.125-0.750:1.00; 0.750-0.875:1.40; 0.875-1.000:20.50",
                "gradual", 8,
            )[0]

        return ([merged],)


class Krea2EditRebalanceSteering:
    """Krea 2 Image Edit with a single steering strength.

    Mirrors ``Krea2EditRebalance``: a single ``steering`` float applies
    ``guidance`` between the refocused main and reference conditionings.
    """

    DEFAULT_MAIN_WEIGHTS = Krea2EditRebalance.DEFAULT_MAIN_WEIGHTS
    DEFAULT_REF_WEIGHTS = Krea2EditRebalance.DEFAULT_REF_WEIGHTS

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP",),
            "steering": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            "layer_multiplier": ("FLOAT", {"default": 1.0, "min": -1000000000.0, "max": 1000000000.0, "step": 0.01}),
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

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("base", "conditioning")

    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/conditioning"

    @staticmethod
    def _batch_len(image):
        return Krea2EditRebalance._batch_len(image)

    @staticmethod
    def _slice_image(image, idx):
        return Krea2EditRebalance._slice_image(image, idx)

    @staticmethod
    def _image_signature(image):
        return Krea2EditRebalance._image_signature(image)

    def main(self, text, clip,
             steering=0.0,
             layer_multiplier=1.0,
             negative=None,
             image1=None, image1_tokens="normal",
             image2=None, image2_tokens="normal",
             image3=None, image3_tokens="normal",
             image4=None, image4_tokens="normal"):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Krea 2 Encode requires ComfyUI (comfy.utils, node_helpers).")

        match_percent = 0.8

        prompt = "" + _align_prompt(text)
        ref_prefix = negative if negative is not None and str(negative) != "" else ""
        prompt_ref = str(ref_prefix) + ""

        image_slots = [
            (image1, image1_tokens),
            (image2, image2_tokens),
            (image3, image3_tokens),
            (image4, image4_tokens),
        ]

        # Determine the pass count from the largest image batch.
        max_batch = 0
        for img, _ in image_slots:
            max_batch = max(max_batch, self._batch_len(img))

        n_passes = max_batch if max_batch > 0 else 1

        guidance_passes = []

        # Per-pass cache: keyed on a signature of the pass's images + params.
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
                float(layer_multiplier),
                prompt, prompt_ref,
            )

            cached = cache.get(key)
            if cached is not None:
                guidance_passes.append(cached)
                continue

            cond_main = compile_edit_krea2(clip, prompt, pass_images if has_image else None)
            cond_ref = compile_edit_krea2(clip, prompt_ref, pass_images if has_image else None)

            # Refocus main and ref with the shared multiplier + fixed layers.
            cond_main = refocus(cond_main, layer_multiplier, self.DEFAULT_MAIN_WEIGHTS)
            cond_ref = refocus(cond_ref, layer_multiplier, self.DEFAULT_REF_WEIGHTS)

            # First guidance for this pass.
            pass_guidance = guidance(cond_main, cond_ref, steering)
            cache.put(key, pass_guidance)
            guidance_passes.append(pass_guidance)

        if not guidance_passes:
            # Fallback: encode text-only.
            final = compile_edit_krea2(clip, prompt, None)
            base = compile_edit_krea2(clip, prompt, None, fixed_res=32)
            return (base, final)

        if len(guidance_passes) == 1:
            merged = guidance_passes[0]
        else:
            # Always fall back to Conditioning Merge (Multi).
            merged = merge_conditioning_multi(guidance_passes, match_percent)

        # Base conditioning encodes the images at a fixed 32px resolution.
        any_image = any(img is not None for img, _ in image_slots)
        base = compile_edit_krea2(clip, prompt, image_slots if any_image else None, fixed_res=32)

        return (base, merged)


NODE_CLASS_MAPPINGS = {
    "ConditioningKrea2Rebalance": ConditioningKrea2Rebalance,
    "Krea2EditRebalance": Krea2EditRebalance,
    "Krea2EditRebalanceSteering": Krea2EditRebalanceSteering,
    "Krea2EncodeRebalance": Krea2EncodeRebalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningKrea2Rebalance": "Conditioning Krea2 Rebalance",
    "Krea2EditRebalance": "Krea 2 Image Edit Rebalance",
    "Krea2EditRebalanceSteering": "Krea 2 Image Edit Rebalance Steering",
    "Krea2EncodeRebalance": "Krea 2 Encode Rebalance",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "ConditioningKrea2Rebalance",
    "Krea2EditRebalance",
    "Krea2EditRebalanceSteering",
    "Krea2EncodeRebalance",
    "KREA2_TAP_LAYERS",
    "KREA2_FEATURE_DIM",
]
