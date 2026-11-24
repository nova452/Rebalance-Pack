"""Krea 2 block-weighted LoRA loader with a frontend "Scan" button.

Scan reads the LoRA's key names (safetensors header only) and reports which
Krea 2 block groups are present; the frontend then shows one 0.0-1.0 float
widget per group found. The group list mirrors ModelMergeKrea2's INPUT_TYPES.
"""

import json
import logging
import re

import folder_paths
import comfy.utils
import comfy.lora


# Block groups, in the same order as ModelMergeKrea2. Each entry matches the
# native model keys after the "diffusion_model." prefix.
_TXT_LAYERWISE = 2
_TXT_REFINER = 2
_MAIN_BLOCKS = 28

KREA2_BLOCK_GROUPS = (
    ["first.", "tmlp.", "txtmlp.", "tproj."]
    + ["txtfusion.layerwise_blocks.{}.".format(i) for i in range(_TXT_LAYERWISE)]
    + ["txtfusion.projector."]
    + ["txtfusion.refiner_blocks.{}.".format(i) for i in range(_TXT_REFINER)]
    + ["blocks.{}.".format(i) for i in range(_MAIN_BLOCKS)]
    + ["last."]
)


def group_widget_name(group):
    """Widget name for a block group ("blocks.3." -> "blocks_3")."""
    return group.rstrip(".").replace(".", "_")


# ---------------------------------------------------------------------------
# Scan: LoRA key names -> block groups.
# ---------------------------------------------------------------------------
_KEY_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")

# Longest first, so ".lora_A.weight" wins over ".lora_A".
_LORA_KEY_SUFFIXES = (
    ".lora_A.weight", ".lora_B.weight", ".lora_down.weight", ".lora_up.weight",
    ".lora_A", ".lora_B", ".lora_down", ".lora_up",
    ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
    ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b",
    ".lokr_w1", ".lokr_w2", ".lokr_t1", ".lokr_t2",
    ".loha_w1_a", ".loha_w1_b", ".loha_w2_a", ".loha_w2_b",
    ".dora_scale", ".alpha", ".diff_b", ".diff", ".set_weight", ".w_norm", ".b_norm",
)

# Trainer module names -> native names. The projections mirror
# comfy.utils.krea2_to_diffusers (krea2_to_diffusers MAP_BASIC) so every form
# ComfyUI's stock loader accepts also lands in the right block group.
_MODULE_RENAMES = (
    (re.compile(r"^transformer_blocks\."), "blocks."),
    (re.compile(r"^text_fusion\."), "txtfusion."),
    (re.compile(r"^final_layer\."), "last."),
    (re.compile(r"^img_in\.?$"), "first."),
    (re.compile(r"^time_embed\.linear_[12]$"), "tmlp."),
    (re.compile(r"^time_embed\.?$"), "tmlp."),
    (re.compile(r"^time_mod_proj$"), "tproj."),
    (re.compile(r"^txt_in\.linear_[123]$"), "txtmlp."),
    (re.compile(r"^txt_in\.?$"), "txtmlp."),
)


def _read_lora_keys(lora_path):
    if lora_path.lower().endswith(".safetensors"):
        try:
            from safetensors import safe_open
        except ImportError:
            logging.warning("Rebalance-Pack Krea2 scan: safetensors unavailable, falling back to a full load.")
        else:
            with safe_open(lora_path, framework="pt") as f:
                return list(f.keys())
    state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
    return list(state_dict.keys())


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
        module = module.replace("_", ".")
    module = _strip_module_suffix(module)
    for pattern, repl in _MODULE_RENAMES:
        module = pattern.sub(repl, module)
    return module


def _find_group(module):
    for group in KREA2_BLOCK_GROUPS:
        # Exact match for leaf modules (first/tmlp/.../projector/last) whose
        # suffix-stripped name has no trailing dot; prefix for block members.
        if module == group[:-1] or module.startswith(group):
            return group
    return None


def scan_krea2_lora_file(lora_path):
    keys = _read_lora_keys(lora_path)
    groups = set()
    modules = set()
    for key in keys:
        module = _normalize_module(key)
        modules.add(module)
        group = _find_group(module)
        if group is not None:
            groups.add(group)
    ordered = [g for g in KREA2_BLOCK_GROUPS if g in groups]
    return ordered, len(modules)


# ---------------------------------------------------------------------------
# Server route: scan a LoRA file from the loras folder by name.
# ---------------------------------------------------------------------------
def _register_krea2_lora_blocks_route():
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        return

    server = PromptServer.instance
    if getattr(server, "_rebalance_krea2_blocks_route_registered", False):
        return
    server._rebalance_krea2_blocks_route_registered = True

    routes = web.RouteTableDef()

    @routes.get("/rebalance_pack/krea2_lora_blocks")
    async def _krea2_lora_blocks(request):
        lora_name = request.query.get("lora_name", "")
        if not lora_name:
            return web.json_response({"error": "Missing lora_name."}, status=400)
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            return web.json_response({"error": "LoRA not found: {}".format(lora_name)}, status=404)
        try:
            groups, modules = scan_krea2_lora_file(lora_path)
        except Exception as ex:
            return web.json_response({"error": "Scan failed: {}".format(ex)}, status=500)
        return web.json_response({"lora": lora_name, "groups": groups, "modules": modules})

    if not server.app.router.frozen:
        try:
            server.app.add_routes(routes)
        except Exception:
            logging.exception("Rebalance-Pack: failed to register /rebalance_pack/krea2_lora_blocks route")


_register_krea2_lora_blocks_route()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class Krea2LoraLoaderBlock:
    """Loads a Krea 2 LoRA with per-block weights.

    "Scan Blocks" reads the selected LoRA and shows a 0.0-1.0 widget for every
    block group found in the file (blocks 0-27, txtfusion blocks, embedders
    first/tmlp/txtmlp/tproj and last). Widget values multiply `strength`.
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
        for group in KREA2_BLOCK_GROUPS:
            required[group_widget_name(group)] = ("FLOAT", dict(block_info))
        required["scan_json"] = ("STRING", {"default": "{}",
                                            "tooltip": "Last scan result, filled by the Scan Blocks button."})
        return {"required": required}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "doit"
    CATEGORY = "loaders"

    DESCRIPTION = (
        "Loads a Krea 2 LoRA with per-block weights. Press Scan Blocks to "
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
        """Per-group ratios from the scan result; None-scale groups stay at 1.0."""
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
        for group in KREA2_BLOCK_GROUPS:
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
            for g in KREA2_BLOCK_GROUPS:
                if k_unet.startswith(g):
                    group = g
                    break

            ratio = 1.0 if scan_ratios is None else scan_ratios.get(group, 1.0)
            if ratio == 0:
                continue
            new_modelpatcher.add_patches({k: v}, strength * ratio)

        return (new_modelpatcher,)


NODE_CLASS_MAPPINGS = {
    "Krea2LoraLoaderBlock": Krea2LoraLoaderBlock,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2LoraLoaderBlock": "Load LoRA Krea2 (Block)",
}

__all__ = [
    "Krea2LoraLoaderBlock",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
