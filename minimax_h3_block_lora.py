"""MiniMax H3 / H3 Image block-weighted LoRA loaders with a frontend "Scan" button.

Scan reads the LoRA's key names (safetensors header only) and reports which
H3 block groups are present; the frontend then shows one 0.0-1.0 float widget
per group found. The group list mirrors the native module layout of
comfy.ldm.minimax.model.MiniMaxH3Model / MiniMaxH3ImageModel. H3 Image is the
same DiT without the audio stream, so its list drops the audio modules.
"""

import json
import logging
import re

import folder_paths
import comfy.utils
import comfy.lora

from .krea2_block_lora import (
    _KEY_PREFIXES,
    _LORA_KEY_SUFFIXES,
    _read_lora_keys,
    group_widget_name,
)


# Block groups, in native-model order. Each entry matches the model keys after
# the "diffusion_model." prefix. The 50 main blocks and 2 refiner blocks are
# the MiniMaxH3Model defaults; time_embedder only exists on non-curve
# checkpoints, but a scan simply won't report it when it is absent.
_MAIN_BLOCKS = 50
_REFINER_BLOCKS = 2


def _build_groups(audio):
    head = ["video_patch_proj."]
    if audio:
        head.append("audio_patch_proj.")
    head += ["condition_proj.", "time_embedder."]
    refiner = ["token_refiner.blocks.{}.".format(i) for i in range(_REFINER_BLOCKS)]
    refiner.append("token_refiner.final_norm.")
    tail = ["final_layer.adaln_proj.", "final_layer.norm.", "final_layer.video_out."]
    if audio:
        tail.append("final_layer.audio_out.")
    return head + refiner + ["blocks.{}.".format(i) for i in range(_MAIN_BLOCKS)] + tail


H3_BLOCK_GROUPS = _build_groups(audio=True)
H3_IMAGE_BLOCK_GROUPS = _build_groups(audio=False)


# ---------------------------------------------------------------------------
# Scan: LoRA key names -> block groups.
# ---------------------------------------------------------------------------
# Trainer module names -> native names (common diffusers-style spellings).
_MODULE_RENAMES = (
    (re.compile(r"^transformer_blocks\."), "blocks."),
    (re.compile(r"^transformer\.blocks\."), "blocks."),
    (re.compile(r"^refiner\.blocks\."), "token_refiner.blocks."),
)

# Kohya underscore-style keys flatten native names that contain underscores
# (video_patch_proj -> video.patch.proj); rejoin the known native modules so
# those keys still land in the right group. Applied in order, all that match.
_DOTTED_TO_NATIVE = (
    ("video.patch.proj", "video_patch_proj"),
    ("audio.patch.proj", "audio_patch_proj"),
    ("condition.proj", "condition_proj"),
    ("time.embedder", "time_embedder"),
    ("token.refiner", "token_refiner"),
    ("final.layer", "final_layer"),
    ("transformer.blocks", "transformer_blocks"),
)
_DOTTED_INNER = (
    ("token_refiner.final.norm", "token_refiner.final_norm"),
    ("final_layer.adaln.proj", "final_layer.adaln_proj"),
    ("final_layer.video.out", "final_layer.video_out"),
    ("final_layer.audio.out", "final_layer.audio_out"),
)


def _strip_module_suffix(module):
    for suffix in _LORA_KEY_SUFFIXES:
        if module.endswith(suffix):
            return module[:-len(suffix)]
    return module


def _normalize_module(key):
    module = key
    for prefix in _KEY_PREFIXES:
        if module.startswith(prefix):
            module = module[len(prefix):]
    if module.startswith("lora_unet_") or module.startswith("lycoris_"):
        module = re.sub(r"^lora_(?:unet|te|te1|te2)_", "", module)
        module = re.sub(r"^lycoris_", "", module)
        # Strip the LoRA tail while the dots are still intact, then flatten.
        module = _strip_module_suffix(module)
        module = module.replace("_", ".")
        for dotted, native in _DOTTED_TO_NATIVE:
            if module.startswith(dotted):
                module = native + module[len(dotted):]
                break
        for dotted, native in _DOTTED_INNER:
            if module.startswith(dotted):
                module = native + module[len(dotted):]
                break
    else:
        module = _strip_module_suffix(module)
    for pattern, repl in _MODULE_RENAMES:
        module = pattern.sub(repl, module)
    return module


def _find_group(module, groups):
    for group in groups:
        # Exact match for leaf modules (video_patch_proj/condition_proj/...)
        # whose suffix-stripped name has no trailing dot; prefix for members.
        if module == group[:-1] or module.startswith(group):
            return group
    return None


def scan_h3_lora_file(lora_path, groups_list):
    keys = _read_lora_keys(lora_path)
    groups = set()
    modules = set()
    for key in keys:
        module = _normalize_module(key)
        modules.add(module)
        group = _find_group(module, groups_list)
        if group is not None:
            groups.add(group)
    ordered = [g for g in groups_list if g in groups]
    return ordered, len(modules)


# ---------------------------------------------------------------------------
# Server route: scan a LoRA file from the loras folder by name.
# ?variant=image scans against the H3 Image group list (no audio modules).
# ---------------------------------------------------------------------------
def _register_h3_lora_blocks_route():
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        return

    server = PromptServer.instance
    if getattr(server, "_rebalance_h3_blocks_route_registered", False):
        return
    server._rebalance_h3_blocks_route_registered = True

    routes = web.RouteTableDef()

    @routes.get("/rebalance_pack/h3_lora_blocks")
    async def _h3_lora_blocks(request):
        lora_name = request.query.get("lora_name", "")
        variant = request.query.get("variant", "h3")
        if not lora_name:
            return web.json_response({"error": "Missing lora_name."}, status=400)
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            return web.json_response({"error": "LoRA not found: {}".format(lora_name)}, status=404)
        groups_list = H3_IMAGE_BLOCK_GROUPS if variant == "image" else H3_BLOCK_GROUPS
        try:
            groups, modules = scan_h3_lora_file(lora_path, groups_list)
        except Exception as ex:
            return web.json_response({"error": "Scan failed: {}".format(ex)}, status=500)
        return web.json_response({"lora": lora_name, "groups": groups, "modules": modules})

    if not server.app.router.frozen:
        try:
            server.app.add_routes(routes)
        except Exception:
            logging.exception("Rebalance-Pack: failed to register /rebalance_pack/h3_lora_blocks route")


_register_h3_lora_blocks_route()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
class _H3LoraLoaderBlockBase:
    GROUPS = H3_BLOCK_GROUPS

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        lora_names = folder_paths.get_filename_list("loras")
        required = {
            "model": ("MODEL",),
            "lora_name": (lora_names,),
            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                                   "tooltip": "Global multiplier on top of every block weight."}),
        }
        block_info = {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                      "tooltip": "Block weight; hidden block groups are not in the scanned LoRA."}
        for group in cls.GROUPS:
            required[group_widget_name(group)] = ("FLOAT", dict(block_info))
        required["scan_json"] = ("STRING", {"default": "{}",
                                            "tooltip": "Last scan result, filled by the Scan Blocks button."})
        return {"required": required}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "doit"
    CATEGORY = "loaders"

    @staticmethod
    def _load_lora(lora_name, cached):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if cached is not None and cached[0] == lora_path:
            lora = cached[1]
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        return lora_path, lora

    def _scan_ratios(self, scan_json, kwargs):
        """Per-group ratios from the scan result; unscanned groups stay at 1.0."""
        try:
            scan = json.loads(scan_json) if scan_json else {}
        except (json.JSONDecodeError, TypeError):
            scan = {}
        groups = scan.get("groups") if isinstance(scan, dict) else None
        scanned = set(groups) if isinstance(groups, list) else None
        if scanned is None:
            return None

        ratios = {}
        for group in scanned:
            widget = kwargs.pop(group_widget_name(group), None)
            try:
                ratios[group] = float(widget) if widget is not None else 1.0
            except (TypeError, ValueError):
                ratios[group] = 1.0
        for group in self.GROUPS:
            ratios.setdefault(group, 1.0)
        return ratios

    def doit(self, model, lora_name, strength=1.0, scan_json="{}", **kwargs):
        if strength == 0:
            return (model,)

        scan_ratios = self._scan_ratios(scan_json, kwargs)

        lora_path, lora = self._load_lora(lora_name, self.loaded_lora)
        self.loaded_lora = (lora_path, lora)

        key_map = comfy.lora.model_lora_keys_unet(model.model)
        loaded = comfy.lora.load_lora(lora, key_map)

        new_modelpatcher = model.clone()

        for k, v in loaded.items():
            # PDD checkpoints map final_layer heads to (key, slice) tuples.
            k_str = k[0] if isinstance(k, tuple) else k
            k_unet = k_str[len("diffusion_model."):] if k_str.startswith("diffusion_model.") else k_str

            group = None
            for g in self.GROUPS:
                if k_unet.startswith(g):
                    group = g
                    break

            ratio = 1.0 if scan_ratios is None else scan_ratios.get(group, 1.0)
            if ratio == 0:
                continue
            new_modelpatcher.add_patches({k: v}, strength * ratio)

        return (new_modelpatcher,)


class MiniMaxH3LoraLoaderBlock(_H3LoraLoaderBlockBase):
    """Loads a MiniMax H3 (video + audio) LoRA with per-block weights.

    "Scan Blocks" reads the selected LoRA and shows a 0.0-1.0 widget for every
    block group found in the file (blocks 0-49, token_refiner blocks, the
    video/audio patch projections, condition_proj, time_embedder and the
    final layer's adaln/norm/video_out/audio_out). Widget values multiply
    `strength`.
    """

    GROUPS = H3_BLOCK_GROUPS

    DESCRIPTION = (
        "Loads a MiniMax H3 LoRA with per-block weights. Press Scan Blocks to "
        "list every block group present in the file as a 0.0-1.0 widget."
    )


class MiniMaxH3ImageLoraLoaderBlock(_H3LoraLoaderBlockBase):
    """Loads a MiniMax H3 Image LoRA with per-block weights.

    Same as the H3 loader without the audio stream modules
    (audio_patch_proj, final_layer.audio_out).
    """

    GROUPS = H3_IMAGE_BLOCK_GROUPS

    DESCRIPTION = (
        "Loads a MiniMax H3 Image LoRA with per-block weights. Press Scan Blocks "
        "to list every block group present in the file as a 0.0-1.0 widget."
    )


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3LoraLoaderBlock": MiniMaxH3LoraLoaderBlock,
    "MiniMaxH3ImageLoraLoaderBlock": MiniMaxH3ImageLoraLoaderBlock,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3LoraLoaderBlock": "Load LoRA H3 (Block)",
    "MiniMaxH3ImageLoraLoaderBlock": "Load LoRA H3 Image (Block)",
}

__all__ = [
    "MiniMaxH3LoraLoaderBlock",
    "MiniMaxH3ImageLoraLoaderBlock",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
