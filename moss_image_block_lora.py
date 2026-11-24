"""MOSS-Image block-weighted LoRA loader with a frontend "Scan" button.

Scan reads the LoRA's key names (safetensors header only) and reports which
MOSS-Image block groups are present; the frontend then shows one 0.0-1.0 float
widget per group found. The group list mirrors the native module layout of
comfy.ldm.moss_image.model.MOSSImage (the granularity a ModelMergeMOSSImage
would expose).
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
# the "diffusion_model." prefix: video_embedder, text_embedder (proj + 2
# blocks), time_embedding, shared_block_modulation, the 42 main layers and the
# three final-stage modules.
_MAIN_LAYERS = 42

MOSS_BLOCK_GROUPS = (
    ["video_embedder.",
     "text_embedder.proj.", "text_embedder.blocks.0.", "text_embedder.blocks.1.",
     "time_embedding.", "shared_block_modulation."]
    + ["layers.{}.".format(i) for i in range(_MAIN_LAYERS)]
    + ["final_norm_video.", "final_linear_video.", "video_head_modulation."]
)


# ---------------------------------------------------------------------------
# Scan: LoRA key names -> block groups.
# ---------------------------------------------------------------------------
# Trainer module names -> native names. MOSS-Image has no diffusers port to
# mirror, so these cover the common trainer spellings of the native modules;
# every form ComfyUI's stock loader accepts also lands in the right group.
_MODULE_RENAMES = (
    (re.compile(r"^transformer_blocks\."), "layers."),
    (re.compile(r"^transformer\.blocks\."), "layers."),
    (re.compile(r"^time_embed(\.(linear_[012]|[012]))?$"), "time_embedding."),
    (re.compile(r"^time\.embed(\.(linear_[012]|[012]))?$"), "time_embedding."),
    (re.compile(r"^(img_in|x_embedder)$"), "video_embedder."),
    (re.compile(r"^(img\.in|x\.embedder)$"), "video_embedder."),
    (re.compile(r"^txt_in$"), "text_embedder.proj."),
    (re.compile(r"^txt\.in$"), "text_embedder.proj."),
)

# Kohya underscore-style keys flatten native names that contain underscores
# (video_embedder -> video.embedder); rejoin the known native modules so those
# keys still land in the right group.
_DOTTED_TO_NATIVE = (
    ("shared.block.modulation", "shared_block_modulation"),
    ("video.head.modulation", "video_head_modulation"),
    ("final.linear.video", "final_linear_video"),
    ("final.norm.video", "final_norm_video"),
    ("video.embedder", "video_embedder"),
    ("text.embedder", "text_embedder"),
    ("time.embedding", "time_embedding"),
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
    else:
        module = _strip_module_suffix(module)
    for dotted, native in _DOTTED_TO_NATIVE:
        if module.startswith(dotted):
            module = native + module[len(dotted):]
            break
    for pattern, repl in _MODULE_RENAMES:
        module = pattern.sub(repl, module)
    return module


def _find_group(module):
    for group in MOSS_BLOCK_GROUPS:
        # Exact match for leaf modules (video_embedder/time_embedding/.../
        # final stage) whose suffix-stripped name has no trailing dot; prefix
        # for block members.
        if module == group[:-1] or module.startswith(group):
            return group
    return None


def scan_moss_image_lora_file(lora_path):
    keys = _read_lora_keys(lora_path)
    groups = set()
    modules = set()
    for key in keys:
        module = _normalize_module(key)
        modules.add(module)
        group = _find_group(module)
        if group is not None:
            groups.add(group)
    ordered = [g for g in MOSS_BLOCK_GROUPS if g in groups]
    return ordered, len(modules)


# ---------------------------------------------------------------------------
# Server route: scan a LoRA file from the loras folder by name.
# ---------------------------------------------------------------------------
def _register_moss_image_lora_blocks_route():
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        return

    server = PromptServer.instance
    if getattr(server, "_rebalance_moss_image_blocks_route_registered", False):
        return
    server._rebalance_moss_image_blocks_route_registered = True

    routes = web.RouteTableDef()

    @routes.get("/rebalance_pack/moss_image_lora_blocks")
    async def _moss_image_lora_blocks(request):
        lora_name = request.query.get("lora_name", "")
        if not lora_name:
            return web.json_response({"error": "Missing lora_name."}, status=400)
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            return web.json_response({"error": "LoRA not found: {}".format(lora_name)}, status=404)
        try:
            groups, modules = scan_moss_image_lora_file(lora_path)
        except Exception as ex:
            return web.json_response({"error": "Scan failed: {}".format(ex)}, status=500)
        return web.json_response({"lora": lora_name, "groups": groups, "modules": modules})

    if not server.app.router.frozen:
        try:
            server.app.add_routes(routes)
        except Exception:
            logging.exception("Rebalance-Pack: failed to register /rebalance_pack/moss_image_lora_blocks route")


_register_moss_image_lora_blocks_route()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class MOSSImageLoraLoaderBlock:
    """Loads a MOSS-Image LoRA with per-block weights.

    "Scan Blocks" reads the selected LoRA and shows a 0.0-1.0 widget for every
    block group found in the file (layers 0-41, text_embedder proj/blocks,
    video_embedder, time_embedding, shared_block_modulation and the final
    stage). Widget values multiply `strength`.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        lora_names = folder_paths.get_filename_list("loras")
        required = {
            "model": ("MODEL",),
            "lora_name": (lora_names,),
            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                                   "tooltip": "Global multiplier on top of every block weight."}),
        }
        block_info = {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                      "tooltip": "Block weight; hidden block groups are not in the scanned LoRA."}
        for group in MOSS_BLOCK_GROUPS:
            required[group_widget_name(group)] = ("FLOAT", dict(block_info))
        required["scan_json"] = ("STRING", {"default": "{}",
                                            "tooltip": "Last scan result, filled by the Scan Blocks button."})
        return {"required": required}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "doit"
    CATEGORY = "loaders"

    DESCRIPTION = (
        "Loads a MOSS-Image LoRA with per-block weights. Press Scan Blocks to "
        "list every block group present in the file as a 0.0-1.0 widget."
    )

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
        for group in MOSS_BLOCK_GROUPS:
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
            k_str = k[0] if isinstance(k, tuple) else k
            k_unet = k_str[len("diffusion_model."):] if k_str.startswith("diffusion_model.") else k_str

            group = None
            for g in MOSS_BLOCK_GROUPS:
                if k_unet.startswith(g):
                    group = g
                    break

            ratio = 1.0 if scan_ratios is None else scan_ratios.get(group, 1.0)
            if ratio == 0:
                continue
            new_modelpatcher.add_patches({k: v}, strength * ratio)

        return (new_modelpatcher,)


NODE_CLASS_MAPPINGS = {
    "MOSSImageLoraLoaderBlock": MOSSImageLoraLoaderBlock,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSImageLoraLoaderBlock": "Load LoRA Moss Image (Block)",
}

__all__ = [
    "MOSSImageLoraLoaderBlock",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
