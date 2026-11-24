import math
import os

import torch

try:
    import comfy.utils
    import node_helpers
    import folder_paths
    _COMFY_AVAILABLE = True
except ImportError:
    _COMFY_AVAILABLE = False


# Encoder profile registry + automatic routing
_ENCODER_PROFILES = {}


class EncoderProfile:

    def __init__(self, name, n_taps, hidden_dim, tap_layers=None):
        self.name = name
        self.n_taps = int(n_taps)
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = self.n_taps * self.hidden_dim
        self.tap_layers = list(tap_layers) if tap_layers is not None else None

    def __repr__(self):
        return "EncoderProfile(name={!r}, n_taps={}, hidden_dim={}, feature_dim={})".format(
            self.name, self.n_taps, self.hidden_dim, self.feature_dim)


def register_encoder_profile(name, n_taps, hidden_dim, tap_layers=None):

    profile = EncoderProfile(name, n_taps, hidden_dim, tap_layers)
    _ENCODER_PROFILES[name] = profile
    # Index by feature_dim for fast lookup
    _ENCODER_PROFILES.setdefault(("_dim", profile.feature_dim), []).append(profile)
    return profile


def get_encoder_profile(name):
    return _ENCODER_PROFILES.get(name)


def detect_encoder_profile(feature_dim):

    matches = _ENCODER_PROFILES.get(("_dim", int(feature_dim)), [])
    if matches:
        return matches[0]
    return None


def _resolve_n_bands(t, default=12):

    flat = t.shape[-1]
    profile = detect_encoder_profile(flat)
    if profile is not None and flat % profile.n_taps == 0:
        return profile.n_taps
    if default > 1 and flat % default == 0:
        return default
    return 1


def _unit_norm_dim(t, eps=1e-8):
    dtype = t.dtype
    t = t.float()
    norm = torch.sqrt(t.pow(2).sum(dim=-1, keepdim=True) + eps)
    return (t / norm).to(dtype)


def _unit_norm_flat(t, eps=1e-8):

    dtype = t.dtype
    t = t.float()
    dims = tuple(range(1, t.dim()))
    norm = torch.sqrt(t.pow(2).sum(dim=dims, keepdim=True) + eps)
    return (t / norm).to(dtype)


def _energy_norm(t, eps=1e-8):

    dims = tuple(range(1, t.dim()))
    return torch.sqrt(t.pow(2).sum(dim=dims, keepdim=True) + eps)


def _match_energy(a, b):

    b = b * (_energy_norm(a) / _energy_norm(b))
    return b


def _split_bands(t, n_bands=None):
    flat = t.shape[-1]
    if n_bands is None:
        n_bands = _resolve_n_bands(t)
    if n_bands > 1 and flat % n_bands == 0:
        d = flat // n_bands
        return t.view(*t.shape[:-1], n_bands, d), d
    return None, None


def _merge_bands(t):
    n_bands = t.shape[-2]
    d = t.shape[-1]
    return t.reshape(*t.shape[:-2], n_bands * d)


def _extract_cond_tensor(item):
    if isinstance(item, (list, tuple)) and len(item) == 2 \
            and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
        return item[0]
    if isinstance(item, torch.Tensor):
        return item
    return None


def _match_batch(ref_dir, target_batch):
    """Match a tensor's batch dimension to ``target_batch``."""
    cur = ref_dir.shape[0]
    if cur == target_batch:
        return ref_dir
    if cur == 1:
        return ref_dir.expand(target_batch, *ref_dir.shape[1:])
    reps = int(math.ceil(target_batch / cur))
    tiled = ref_dir.repeat(reps, *([1] * (ref_dir.dim() - 1)))
    return tiled[:target_batch]


def _common_batch(tensors):
    return max(t.shape[0] for t in tensors)


def clip_signature(clip):
    """Cheap identity of a CLIP object for pass caching.

    Combines object id with the parameter count of the text model so that
    swapping checkpoints invalidates the cache (object id alone could in
    principle be reused after garbage collection)."""
    if clip is None:
        return ("none",)
    params = -1
    try:
        params = sum(p.numel() for p in clip.cond_stage_model.parameters())
    except Exception:
        pass
    return (id(clip), type(clip).__name__, params)


def image_signature(image):
    """Signature for caching: shape + a hash of the pixels. Shape alone is not enough"""
    if image is None:
        return ("none",)
    if isinstance(image, list):
        return ("list", len(image),
                tuple(image_signature(img) for img in image))
    if hasattr(image, "shape"):
        try:
            import hashlib
            data = image.detach().cpu().contiguous().numpy().tobytes()
            return ("tensor", tuple(image.shape),
                    hashlib.sha1(data).hexdigest())
        except Exception:
            # Fall back to shape-only rather than failing the encode.
            return ("tensor", tuple(image.shape))
    return ("unknown",)


class PassCache:
    """Replaces the old unbounded instance dict"""

    def __init__(self, max_entries=32):
        self.max_entries = int(max_entries)
        self._data = {}

    def get(self, key):
        if key in self._data:
            self._data[key] = self._data.pop(key)  # refresh recency
            return self._data[key]
        return None

    def put(self, key, value):
        self._data[key] = value
        while len(self._data) > self.max_entries:
            self._data.pop(next(iter(self._data)))


# Module-level encode cache: identical (clip, prompt, images, template,
# positions) encodes return the cached conditioning instead of re-running the
# encoder. This makes repeated queue runs deterministic — GPU kernels are
# nondeterministic, so re-encoding the same input yields slightly different
# hidden states each time. Entries hold CPU copies so VRAM is not pinned.
_ENCODE_CACHE = PassCache(max_entries=256)


def _align_prompt(text):
    if text is None:
        return ""
    return str(text).replace("{", "{").replace("}", "}")


def _parse_floats(s):
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        vals = [float(x) for x in s.replace(";", ",").split(",") if x.strip() != ""]
    except ValueError:
        return None
    if len(vals) < 2:
        return None
    return vals

# ignored
SYS_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# Target longest-side resolution per tier. Each image is scaled to it selected resolution independently
RESOLUTIONS = {"low": 384, "normal": 768, "high": 1024, "max": 1280, "focal": 32}


def _scale_to_resolution(samples, target):

    n, c, h, w = samples.shape
    if h == target and w == target:
        return samples
    scale = target / max(h, w)
    nh = max(1, round(h * scale))
    nw = max(1, round(w * scale))
    return comfy.utils.common_upscale(samples, nw, nh, "area", "disabled")


def _apply_image_positions(position_ids, embeds_info, positions):
    """Positions, ignored."""
    
    imgs = sorted([e for e in embeds_info if e.get("type") == "image"],
                  key=lambda e: e["index"])
    for e, pos in zip(imgs, positions):
        if pos is None:
            continue
        x0, y0 = pos
        grid = e["extra"]["grid"]
        start, size = e["index"], e["size"]
        end = start + size
        gh = max(1, int(grid[0][1]) // 2)
        gw = max(1, int(grid[0][2]) // 2)
        device = position_ids.device
        h_ids = (y0 + torch.arange(gh, device=device)).unsqueeze(1).expand(gh, gw).reshape(-1)
        w_ids = (x0 + torch.arange(gw, device=device)).unsqueeze(0).expand(gh, gw).reshape(-1)
        position_ids[1, start:end] = h_ids[:end - start]
        position_ids[2, start:end] = w_ids[:end - start]


def _qwen_image_positions_ctx(clip, positions):
    """Patch a Qwen3-VL text encoder's ``build_image_inputs`` during encoding.

    Returns ``(enter, exit)`` callables. ``enter`` swaps in a wrapper that
    applies ``positions`` (list of ``(x0, y0)`` or ``None`` per image, in
    input order) to the computed MRoPE position ids; ``exit`` restores the
    original method. No-op when the encoder isn't Qwen3-VL or no positions
    are set."""
    positions = [p for p in (positions or []) if p is not None]
    tr = None
    try:
        for m in clip.cond_stage_model.modules():
            if hasattr(m, "build_image_inputs") and hasattr(m, "visual"):
                tr = m
                break
    except AttributeError:
        tr = None
    if tr is None or not positions:
        return (lambda: None), (lambda: None)

    orig = tr.build_image_inputs

    def patched(embeds, embeds_info):
        position_ids, masks, deepstack = orig(embeds, embeds_info)
        if position_ids is not None:
            _apply_image_positions(position_ids, embeds_info, positions)
        return position_ids, masks, deepstack

    def enter():
        tr.build_image_inputs = patched

    def exit_():
        tr.build_image_inputs = orig

    return enter, exit_


def _capture_image_spans(clip):
    """Patch ``build_image_inputs`` during encoding to record image spans.

    Returns ``(enter, exit_, spans)``. After ``exit_()`` runs, ``spans`` holds
    a list of ``(start, size)`` tuples — one per image, in input order —
    locating each image's token span inside the encoded sequence. No-op
    (empty list) when the encoder isn't a Qwen-VL style model."""
    spans = []
    tr = None
    try:
        for m in clip.cond_stage_model.modules():
            if hasattr(m, "build_image_inputs") and hasattr(m, "visual"):
                tr = m
                break
    except AttributeError:
        tr = None
    if tr is None:
        return (lambda: None), (lambda: None), spans

    orig = tr.build_image_inputs

    def patched(embeds, embeds_info):
        for e in sorted([e for e in embeds_info if e.get("type") == "image"],
                        key=lambda e: e["index"]):
            spans.append((e["index"], e["size"]))
        return orig(embeds, embeds_info)

    def enter():
        tr.build_image_inputs = patched

    def exit_():
        tr.build_image_inputs = orig

    return enter, exit_, spans


def _qwen_image_spans(conditioning):
    """Read image spans recorded by ``_capture_image_spans`` from extras.

    Returns a list of ``(start, size)`` tuples, or ``[]`` when the
    conditioning carries no span info (e.g. plain text encode)."""
    spans = []
    for item in (conditioning or []):
        if isinstance(item, (list, tuple)) and len(item) == 2 \
                and isinstance(item[1], dict):
            s = item[1].get("image_spans")
            if isinstance(s, (list, tuple)):
                spans.extend([tuple(x) for x in s])
    return spans


def _image_span_mask(seq_len, spans, device):
    """Boolean mask ``(1, seq_len)`` marking image-token positions."""
    m = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    for start, size in spans:
        start = max(0, int(start))
        end = min(seq_len, start + max(0, int(size)))
        if end > start:
            m[0, start:end] = True
    return m


def _blend_image_spans(cond_a, cond_b, spans_a, spans_b, mode="average",
                       strength=0.5, match_percent=0.5):
    """Blend the image-token spans of two Qwen-VL conditionings.

    Text tokens come from ``cond_a`` unchanged. Image tokens are blended
    between the two encodings using the requested strategy, mirroring the
    IPAdapter ``Combine Embeds`` methods but operating on the interleaved
    image spans of a joint text+image sequence.

    ``mode``:
      average      element-wise mean of the two spans
      norm_average energy-matched mean (neither image dominates by magnitude)
      weighted     ``a * (1 - strength) + b * strength``
      norm_weighted  energy-matched weighted blend
      add          ``a + b``
      subtract     ``a - b``
      max_magnitude  keep the element with larger magnitude
      min_magnitude  keep the element with smaller magnitude
      top_match    average where both are strong and agree in sign,
                   otherwise keep the dominant element (Rebalance default)
      concat       interleave: keep both spans back to back (doubles the
                   image token count; positions are not re-mapped, so this
                   is best used with a single reference image)

    Spans are blended per-position: each span of ``a`` is paired with the
    same-index span of ``b`` (matched by order, truncated to the shorter
    length). Batch dims are matched by broadcast/tiling first."""
    orig_dtype = cond_a.dtype
    a = cond_a.float()
    b = cond_b.float()

    target_batch = _common_batch([a, b])
    a = _match_batch(a, target_batch)
    b = _match_batch(b, target_batch)

    out = a.clone()

    n = min(len(spans_a), len(spans_b))
    if n == 0:
        # No span info on one side: fall back to blending the whole sequence.
        blended = _merge_tensor_mode(a, b, mode, match_percent=match_percent,
                                     strength=strength)
        return blended.to(orig_dtype)

    for i in range(n):
        sa, sb = spans_a[i], spans_b[i]
        start_a, size_a = int(sa[0]), int(sa[1])
        start_b, size_b = int(sb[0]), int(sb[1])
        size = min(size_a, size_b)
        if size <= 0:
            continue
        seg_a = a[:, start_a:start_a + size]
        seg_b = b[:, start_b:start_b + size]
        blended = _merge_tensor_mode(seg_a, seg_b, mode,
                                     match_percent=match_percent,
                                     strength=strength)
        out[:, start_a:start_a + size] = blended

    return out.to(orig_dtype)


def merge_image_embeds(clip, conditioning_a, conditioning_b, mode="average",
                       strength=0.5, match_percent=0.5):
    """Blend the image spans of two Qwen-VL conditionings (see
    ``_blend_image_spans``). Requires both conditionings to have been encoded
    with span capture (``QwenVLImageEmbedsEncode``); falls back to a
    whole-sequence blend when span info is missing."""
    a_t, spans_a = _structure_tensor_and_spans(conditioning_a)
    b_t, spans_b = _structure_tensor_and_spans(conditioning_b)
    if a_t is None or b_t is None:
        return conditioning_a

    out = []
    items_a = list(conditioning_a if isinstance(conditioning_a, (list, tuple)) else [conditioning_a])
    b_items = list(conditioning_b if isinstance(conditioning_b, (list, tuple)) else [conditioning_b])
    # Per-item tensors for the write-back loop (concatenated tensor is only
    # used for the early guard above).
    for idx, item in enumerate(items_a):
        if isinstance(item, (list, tuple)) and len(item) == 2 \
                and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
            cond_t, extras = item
            b_item = b_items[idx] if idx < len(b_items) else None
            b_t_item = _extract_cond_tensor(b_item) if b_item is not None else None
            if b_t_item is not None:
                new_t = _blend_image_spans(cond_t, b_t_item, spans_a, spans_b,
                                           mode=mode, strength=strength,
                                           match_percent=match_percent)
                new_extras = dict(extras)
                # Keep span metadata valid: spans now describe the blended
                # image tokens at a's positions.
                new_extras["image_spans"] = [list(s) for s in spans_a]
                out.append([new_t, new_extras])
            else:
                out.append([cond_t, dict(extras)])
        else:
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Batch reducers for style extraction from many image conditionings.
#
# Unlike pairwise merges, these compute statistics over the *whole* batch of
# image-token spans at once, so each reducer produces a genuinely different
# style signature. All operate per image-span (spans from the first
# conditioning define the output layout; other inputs' spans are aligned by
# index and truncated to the shortest span length).
# ---------------------------------------------------------------------------

_STYLE_REDUCERS = [
    "mean",          # centroid of all images — the classic style average
    "median",        # per-element median — robust to outlier images
    "trimmed_mean",  # mean after dropping the most extreme `strength` fraction
    "consensus",     # keep elements where most images agree in sign (strength = agreement ratio)
    "dominant",      # per-element value from the image with max magnitude there
    "spread",        # mean + strength * per-element std — amplifies what varies across images
    "centroid_dir",  # unit-norm mean direction, rescaled to the batch's mean energy
    # Batch-level versions of the classic merge modes (first input is the
    # anchor, the rest of the batch is the "other" side):
    "average",       # alias of mean
    "norm_average",  # energy-matched mean — every image rescaled to the anchor's energy first
    "weighted",      # anchor * (1 - strength) + mean(rest) * strength
    "norm_weighted", # weighted, with mean(rest) energy-matched to the anchor
    "add",           # sum of all images (scales with batch size)
    "subtract",      # anchor - mean(rest) — what the anchor has that the batch doesn't
    "max_magnitude", # alias of dominant
    "min_magnitude", # per-element value from the image with min magnitude there
    "top_match",     # average where most images are strong and agree in sign, else dominant (strength = match_percent)
    "concat",        # keep every image's span as separate tokens, stacked back to back (multi-reference, not style extraction)
    # Additional reducers:
    "geometric_mean",  # exp(mean(log|x|)) * sign(mean) — multiplicative style: features present in ALL images survive
    "harmonic_mean",   # N / sum(1/x) — punishes features that are weak in any image
    "rms",             # root-mean-square — energy-preserving centroid, brighter than mean
    "mode_stats",      # most frequent sign per element with mean magnitude of the agreeing images
    "pca_style",       # first principal component of the batch — the dominant axis of variation
    "outlier",         # the image furthest from the batch mean (strength = rank: 0 = most typical, 1 = most extreme)
    "random",          # pick one image at random (seeded by strength * 1000)
]

# Modes that ignore the strength slider entirely.
_STRENGTH_FREE = {
    "mean", "median", "dominant", "centroid_dir",
    "average", "norm_average", "add", "subtract",
    "max_magnitude", "min_magnitude", "concat",
    "geometric_mean", "harmonic_mean", "rms", "mode_stats", "pca_style",
}


def _item_spans(item):
    """Read image spans from a single conditioning item's extras."""
    if isinstance(item, (list, tuple)) and len(item) == 2 \
            and isinstance(item[1], dict):
        s = item[1].get("image_spans")
        if isinstance(s, (list, tuple)):
            return [tuple(x) for x in s]
    return []


def _structure_tensor_and_spans(s):
    """Extract the sequence tensor and image spans from one conditioning
    structure (a list of ``[tensor, dict]`` pairs).

    Returns ``(tensor, spans)`` or ``(None, [])``. Multiple items are
    concatenated along the token dimension (spans are relative to the full
    sequence)."""
    items = s if isinstance(s, (list, tuple)) else [s]
    ts = []
    spans = []
    for item in items:
        t = _extract_cond_tensor(item)
        if t is None:
            continue
        ts.append(t)
        spans.extend(_item_spans(item))
    if not ts:
        return None, []
    tensor = ts[0] if len(ts) == 1 else torch.cat(ts, dim=1)
    return tensor, spans


def _collect_span_tensors(structures):
    """Extract per-structure image-span tensors for batch reduction.

    Returns ``(spans, stack)`` where ``spans`` is the span list of the first
    structure (output layout) and ``stack`` is a tensor ``(N, B, T, D)`` of
    the aligned span segments from every structure (truncated to the shortest
    span, batch-matched to the first). Structures without span info
    contribute their full sequence, aligned by token position."""
    tensors = []
    ref_spans = None
    ref_batch = None
    for s in structures:
        t, spans = _structure_tensor_and_spans(s)
        if t is None:
            continue
        if ref_spans is None:
            ref_spans = spans
            ref_batch = t.shape[0]
        tensors.append((t, spans))

    if not tensors:
        return [], None

    # Determine the per-index segment length: min across structures.
    n_spans = min(len(spans) for _, spans in tensors) if all(tensors[i][1] for i in range(len(tensors))) else 0
    if n_spans == 0:
        # Fall back to whole-sequence alignment: truncate all to the shortest
        # sequence and treat it as one span.
        seq = min(t.shape[1] for t, _ in tensors)
        segs = [_match_batch(t[:, :seq].float(), ref_batch) for t, _ in tensors]
        return [(0, seq)], torch.stack(segs, dim=0)

    stack = []
    for t, spans in tensors:
        parts = []
        for i in range(n_spans):
            start, size = int(spans[i][0]), int(spans[i][1])
            parts.append(t[:, start:start + size].float())
        stack.append(torch.cat(parts, dim=1))
    min_len = min(s.shape[1] for s in stack)
    stack = [_match_batch(s[:, :min_len], ref_batch) for s in stack]
    # Report spans in the first structure's layout, truncated to min_len.
    out_spans = []
    off = 0
    for i in range(n_spans):
        size = int(ref_spans[i][1])
        size = max(0, min(size, min_len - off))
        if size <= 0:
            break
        out_spans.append((int(ref_spans[i][0]), size))
        off += size
        if off >= min_len:
            break
    return out_spans, torch.stack(stack, dim=0)


def _reduce_style_tensor(stack, mode, strength, harmonic_sign="clamp_positive"):
    """Reduce a ``(N, B, T, D)`` stack of span tensors to ``(B, T, D)``."""
    n = stack.shape[0]
    if n == 1:
        return stack[0]
    s = float(min(max(strength, 0.0), 1.0))

    if mode == "median":
        return stack.median(dim=0).values

    if mode == "trimmed_mean":
        # Drop the most extreme fraction of images per element (by distance
        # from the median), then average the rest. `strength` = fraction to
        # drop from each side (0 = plain mean, 0.5 = median).
        keep = n - 2 * int(round(s * n / 2.0))
        keep = max(1, min(n, keep))
        if keep >= n:
            return stack.mean(dim=0)
        med = stack.median(dim=0, keepdim=True).values
        dist = (stack - med).abs().sum(dim=(-1, -2), keepdim=True)
        order = dist.argsort(dim=0)  # (N, B, 1, 1)
        sel = order[:keep]
        return stack.gather(0, sel.expand(sel.shape[:1] + stack.shape[1:])).mean(dim=0)

    if mode == "consensus":
        # Keep an element only where enough images agree in sign; where they
        # disagree, shrink toward zero (cancellation) instead of averaging
        # conflicting features. `strength` = required agreement ratio.
        ratio = 0.5 + s * 0.5  # 0.5..1.0
        signs = stack.sign()
        pos_frac = (signs > 0).float().mean(dim=0)
        neg_frac = (signs < 0).float().mean(dim=0)
        agree = torch.maximum(pos_frac, neg_frac)
        mean = stack.mean(dim=0)
        return mean * (agree >= ratio).float() * agree.clamp(min=0.0)

    if mode == "dominant":
        # Per element, take the value from the image whose magnitude there is
        # largest — a "strongest feature wins" signature.
        idx = stack.abs().argmax(dim=0, keepdim=True)
        return stack.gather(0, idx.expand(1, *stack.shape[1:])).squeeze(0)

    if mode == "spread":
        # mean + strength * std: keeps the style centroid but amplifies the
        # directions in which the images actually differ — good for turning
        # a diverse set into a "characterful" style rather than a mushy mean.
        mean = stack.mean(dim=0)
        if s <= 0.0:
            return mean
        std = stack.std(dim=0)
        return mean + s * std

    if mode == "centroid_dir":
        # Unit-norm mean direction (pure style direction, magnitude-free),
        # rescaled to the batch's mean per-token energy.
        direction = _unit_norm_flat(stack.mean(dim=0))
        energy = _energy_norm(stack).mean(dim=0)
        return direction * energy

    # Batch-level versions of the classic merge modes. The first image is
    # the anchor; the rest of the batch is reduced to mean(rest) first, so
    # these stay distinct no matter how large the batch is.
    anchor = stack[0]
    rest = stack[1:].mean(dim=0) if n > 1 else anchor

    if mode == "norm_average":
        # Energy-match every image to the anchor before averaging, so no
        # single high-energy image dominates the centroid.
        matched = torch.stack(
            [_match_energy(anchor, stack[i]) for i in range(n)], dim=0)
        return matched.mean(dim=0)

    if mode == "weighted":
        return anchor * (1.0 - s) + rest * s

    if mode == "norm_weighted":
        return anchor * (1.0 - s) + _match_energy(anchor, rest) * s

    if mode == "add":
        return stack.sum(dim=0)

    if mode == "subtract":
        # anchor - mean(rest): keeps only what the anchor has that the rest
        # of the batch doesn't — a "difference from the crowd" signature.
        return anchor - rest

    if mode == "min_magnitude":
        idx = stack.abs().argmin(dim=0, keepdim=True)
        return stack.gather(0, idx.expand(1, *stack.shape[1:])).squeeze(0)

    if mode == "top_match":
        # Elements where most images are in the top `s` fraction by magnitude
        # AND agree in sign get averaged; everywhere else the dominant
        # image's value wins. `strength` = match_percent (0..1).
        # Sort-based threshold: torch.quantile has an element-count limit on
        # CUDA that large spans can exceed.
        mp = s
        masks = []
        for x in stack:
            if mp < 1.0:
                mag_flat = x.abs().flatten().float()
                k = max(1, int(round(mp * mag_flat.numel())))
                thresh = mag_flat.sort(descending=True).values[k - 1]
                masks.append(x.abs() >= thresh)
            else:
                masks.append(torch.ones_like(x, dtype=torch.bool))
        masks = torch.stack(masks, dim=0)
        signs = torch.stack([x.sign() for x in stack], dim=0)
        match = masks.all(dim=0) & (signs == signs[0:1]).all(dim=0)
        avg = stack.mean(dim=0)
        idx = stack.abs().argmax(dim=0, keepdim=True)
        fallback = stack.gather(0, idx.expand(1, *stack.shape[1:])).squeeze(0)
        return torch.where(match, avg, fallback)

    if mode == "max_magnitude":
        return stack.gather(0, stack.abs().argmax(dim=0, keepdim=True)
                            .expand(1, *stack.shape[1:])).squeeze(0)

    if mode == "average":
        return stack.mean(dim=0)

    if mode == "geometric_mean":
        # exp(mean(log|x|)) with the batch-mean sign: a feature survives only
        # if it is consistently present across ALL images — multiplicative
        # consensus, much stricter than the arithmetic mean.
        mag = stack.abs().clamp_min(1e-8)
        gm = torch.exp(torch.log(mag).mean(dim=0))
        sign = stack.mean(dim=0).sign()
        return gm * sign

    if mode == "harmonic_mean":
        # N / sum(1/x): punishes any feature that is weak in even one image —
        # the "must be present everywhere" centroid.
        #
        # ``harmonic_sign`` controls how negative values are treated:
        #   clamp_positive — clamp negatives up to +eps (legacy behavior):
        #     any feature negative in even ONE image is crushed toward zero,
        #     acting as a strict positive-consensus filter.
        #   signed — keep the sign: tiny magnitudes of either sign dominate
        #     the reciprocal, so features near zero in any image (either
        #     sign) are suppressed, but consistently negative features
        #     survive as negative.
        if harmonic_sign == "signed":
            safe = torch.where(stack.abs() < 1e-8,
                               torch.sign(stack) * 1e-8,
                               stack)
        else:
            safe = stack.clamp_min(1e-8)
        return n / (1.0 / safe).sum(dim=0)

    if mode == "rms":
        # Root-mean-square: energy-preserving centroid. Unlike the mean it
        # never cancels opposing signs, so it stays "bright" where images
        # disagree.
        return stack.pow(2).mean(dim=0).sqrt()

    if mode == "mode_stats":
        # Most frequent sign per element; magnitude is the mean of the images
        # that share that sign. Disagreeing images are excluded rather than
        # averaged in.
        signs = stack.sign()
        pos_frac = (signs > 0).float().mean(dim=0)
        neg_frac = (signs < 0).float().mean(dim=0)
        pos_w = pos_frac / (pos_frac + neg_frac).clamp_min(1e-8)
        take_pos = pos_w >= 0.5
        mag = stack.abs().mean(dim=0)
        pos_mean = torch.where(stack > 0, stack.abs(), torch.zeros_like(stack)).sum(dim=0) / pos_frac.clamp_min(1e-8) / n
        neg_mean = torch.where(stack < 0, stack.abs(), torch.zeros_like(stack)).sum(dim=0) / neg_frac.clamp_min(1e-8) / n
        return torch.where(take_pos, pos_mean, -neg_mean) * (mag > 0).float()

    if mode == "pca_style":
        # First principal component across the N images, computed per token
        # position in feature space (vectorized power iteration — no SVD).
        # Output: the dominant variation direction, scaled per token by the
        # mean absolute projection of the images onto it (how much style
        # variation actually lives along that axis).
        n_, b_, t_, d_ = stack.shape
        outs = []
        for b in range(b_):
            X = stack[:, b]                                   # (N, T, D)
            Xc = X - X.mean(dim=0, keepdim=True)              # center across images
            V = torch.randn(t_, d_, device=X.device, dtype=X.dtype)
            V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            for _ in range(8):
                U = torch.einsum('ntd,td->nt', Xc, V)         # (N, T) scores
                V = torch.einsum('ntd,nt->td', Xc, U)         # (T, D) direction
                V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            energy = torch.einsum('ntd,td->nt', Xc, V).abs().mean(dim=0)  # (T,)
            outs.append((V * energy.unsqueeze(-1)).unsqueeze(0))
        return torch.cat(outs, dim=0).to(stack.dtype)         # (B, T, D)

    if mode == "outlier":
        # Pick the image whose distance from the batch mean ranks at
        # `strength`: 0 = the most typical image, 1 = the most extreme
        # outlier. Useful to grab "the most representative" or "the most
        # characterful" single image from the batch.
        mean = stack.mean(dim=0, keepdim=True)
        dist = (stack - mean).pow(2).sum(dim=(-1, -2))  # (N, B)
        order = dist.argsort(dim=0)  # typical -> extreme
        rank = int(round(s * (n - 1)))
        rank = max(0, min(n - 1, rank))
        idx = order[rank]  # (B,)
        sel = stack.gather(0, idx.view(1, -1, 1, 1).expand(1, *stack.shape[1:])).squeeze(0)
        return sel

    if mode == "random":
        # Deterministic random pick: strength * 1000 acts as the seed, so
        # the same strength always picks the same image.
        seed = int(round(s * 1000))
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        idx = torch.randint(0, n, (1,), generator=g).item()
        return stack[idx]

    # default: mean
    return stack.mean(dim=0)


def _reduce_style_concat(stack, spans):
    """Concat reducer: keep every image's span as separate tokens, stacked
    back to back — the analogue of IPAdapter Combine Embeds' ``concat``,
    where the batch dimension grows instead of the images being averaged.

    Returns ``(reduced, out_spans)``: the concatenated tensor ``(B, N*T, D)``
    and the span list describing each image's segment inside it (all at the
    same start, sequential in the token dim)."""
    n = stack.shape[0]
    parts = [stack[i] for i in range(n)]  # each (B, T, D)
    reduced = torch.cat(parts, dim=1)     # (B, N*T, D)
    total = reduced.shape[1]
    per = total // n if n else total
    out_spans = [(0, per)]                # one span covering the whole block
    return reduced, out_spans


# ---------------------------------------------------------------------------
# Style metrics: quantify how strong / coherent the merged style is, and
# amplify it. All metrics are computed on the (N, B, T, D) span stack and the
# reduced (B, T, D) tensor, in float32, and averaged over batch items.
# ---------------------------------------------------------------------------

def compute_style_metrics(stack, reduced):
    """Quantify the merged style.

    Returns a dict of scalar floats:
      coherence   mean cosine similarity of each image's span to the reduced
                  style (1.0 = all images identical, low = diverse batch)
      energy      L2 norm of the reduced style per token (absolute strength
                  of the signal the model will attend to)
      contrast    std across images along the reduced direction, relative to
                  its energy (how much the images vary around the style)
      distinct    distance of the reduced style from the *text* baseline is
                  not available here; instead: fraction of elements whose
                  magnitude exceeds 2x the batch mean magnitude (how peaked /
                  selective the style signature is)
      coverage    fraction of elements that survived reduction with
                  meaningful magnitude (low = sparse, selective signature)
    """
    n = stack.shape[0]
    b = stack.shape[1]
    red = reduced.float()

    # Per-batch-item metrics, then averaged.
    coh_list, energy_list, contrast_list, distinct_list, coverage_list = [], [], [], [], []
    for bi in range(b):
        X = stack[:, bi].float()          # (N, T, D)
        r = red[bi]                       # (T, D)

        # Coherence: cosine similarity of each image to the reduced style.
        rn = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        xn = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos = (xn * rn).sum(dim=-1)       # (N, T)
        coh_list.append(cos.mean().item())

        energy_list.append(r.norm(dim=-1).mean().item())

        # Contrast: projection std of the images onto the style direction,
        # normalized by style energy.
        proj = (X * rn).sum(dim=-1)       # (N, T)
        contrast_list.append((proj.std(dim=0) / proj.abs().mean(dim=0).clamp_min(1e-8)).mean().item())

        # Distinctiveness: fraction of elements > 2x batch mean magnitude.
        mag = X.abs().mean(dim=0)         # (T, D) mean magnitude per element
        r_mag = r.abs()
        distinct_list.append((r_mag > 2.0 * mag.clamp_min(1e-8)).float().mean().item())

        # Coverage: fraction of elements with meaningful magnitude relative
        # to the style's own scale.
        scale = r_mag.mean().clamp_min(1e-8)
        coverage_list.append((r_mag > 0.1 * scale).float().mean().item())

    return {
        "coherence": sum(coh_list) / b,
        "energy": sum(energy_list) / b,
        "contrast": sum(contrast_list) / b,
        "distinct": sum(distinct_list) / b,
        "coverage": sum(coverage_list) / b,
    }


def apply_style_boost(reduced, stack, boost):
    """Amplify the merged style signal.

    ``boost`` in 0..20. At 0 the reduced style is returned unchanged. The
    amplification is *adaptive*: the gain per token is driven by how much the
    images agree along each direction, so shared style features (high
    coherence) get boosted hard while noisy, image-specific features are left
    alone. This is what makes the style "strong" without blowing the whole
    conditioning up.

    gain = 1 + boost * agreement, where agreement is the per-element fraction
    of images whose projection onto the reduced direction shares its sign
    (0..1). Energy is renormalized so the boost changes *directional
    emphasis*, not overall loudness — the sampler is sensitive to absolute
    energy, and uncontrolled scaling just shifts colors."""
    if boost <= 0.0:
        return reduced
    boost = float(min(boost, 20.0))
    n, b = stack.shape[0], stack.shape[1]
    out = reduced.clone()
    for bi in range(b):
        X = stack[:, bi].float()          # (N, T, D)
        r = reduced[bi].float()           # (T, D)
        rn = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        proj = (X * rn).sum(dim=-1)       # (N, T) projections onto style dir
        # Agreement: fraction of images whose projection has the same sign as
        # the mean projection (i.e. the style direction is consistent).
        mean_proj = proj.mean(dim=0, keepdim=True)          # (1, T)
        agree = ((proj.sign() == mean_proj.sign()).float().mean(dim=0))  # (T,)
        gain = 1.0 + boost * agree        # (T,)
        boosted = r * gain.unsqueeze(-1)
        # Renormalize energy to the original per-token norm.
        orig_norm = r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        new_norm = boosted.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out[bi] = (boosted * (orig_norm / new_norm)).to(reduced.dtype)
    return out


def reduce_image_embeds(clip, structures, mode="mean", strength=0.5,
                        harmonic_sign="clamp_positive",
                        style_boost=0.0, return_metrics=False):
    """Reduce many Qwen-VL conditionings to a single style conditioning.

    ``structures`` is a list of conditioning structures (as produced by
    ``QwenVLImageEmbedsEncode``). Text tokens come from the first structure
    unchanged; the image-token spans of all structures are reduced with the
    requested batch strategy (see ``_STYLE_REDUCERS``). The result keeps
    ``image_spans`` metadata so it can be re-reduced or blended further.

    ``harmonic_sign`` only affects the ``harmonic_mean`` mode — see
    ``_reduce_style_tensor``.

    ``style_boost`` adaptively amplifies the merged style (see
    ``apply_style_boost``). ``return_metrics`` additionally returns a dict of
    scalar style metrics (see ``compute_style_metrics``)."""
    structures = [s for s in (structures or []) if s is not None]
    if not structures:
        raise ValueError("reduce_image_embeds: at least one conditioning is required.")
    if len(structures) == 1:
        if return_metrics:
            spans0, stack0 = _collect_span_tensors(structures)
            if stack0 is not None:
                # stack0 is (1, B, T, D); reduce trivially to get the
                # "merged" tensor for metrics without touching the output.
                metrics = compute_style_metrics(stack0, stack0[0])
            else:
                metrics = {}
            return structures[0], metrics
        return structures[0]

    spans, stack = _collect_span_tensors(structures)
    if stack is None:
        if return_metrics:
            return structures[0], {}
        return structures[0]

    if mode == "concat":
        # Multi-reference: every image's span becomes separate tokens stacked
        # back to back. The output sequence grows to text + N*T image tokens,
        # so it cannot be written back into the first structure's layout —
        # a new conditioning is built from the first structure's text tokens
        # followed by the concatenated image block.
        reduced, out_spans = _reduce_style_concat(stack, spans)  # (B, N*T, D)
        reduced = apply_style_boost(reduced, stack, style_boost)
        metrics = compute_style_metrics(stack, reduced) if return_metrics else None
        base = structures[0]
        items = list(base if isinstance(base, (list, tuple)) else [base])
        out = []
        for item in items:
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                # Replace the first structure's image span(s) with the
                # concatenated block; text before the span is kept.
                first_start = int(spans[0][0]) if spans else cond_t.shape[1]
                prefix = cond_t[:, :first_start]
                new_t = torch.cat([prefix, reduced.to(cond_t.dtype)], dim=1)
                new_extras = dict(extras)
                new_extras["image_spans"] = [[first_start, reduced.shape[1]]]
                out.append([new_t, new_extras])
            else:
                out.append(item)
        return (out, metrics) if return_metrics else out

    reduced = _reduce_style_tensor(stack, mode, strength,
                                   harmonic_sign=harmonic_sign)  # (B, T', D)
    reduced = apply_style_boost(reduced, stack, style_boost)
    metrics = compute_style_metrics(stack, reduced) if return_metrics else None

    base = structures[0]
    out = []
    items = list(base if isinstance(base, (list, tuple)) else [base])
    seg_iter = 0
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2 \
                and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
            cond_t, extras = item
            new_t = cond_t.clone()
            # Write the reduced segments back into the span positions.
            off = 0
            for start, size in spans:
                seg = reduced[:, off:off + size]
                end = min(cond_t.shape[1], int(start) + int(size))
                seg_len = end - int(start)
                if seg_len <= 0:
                    break
                new_t[:, int(start):end] = seg[:, :seg_len].to(new_t.dtype)
                off += size
            new_extras = dict(extras)
            new_extras["image_spans"] = [list(s) for s in spans]
            out.append([new_t, new_extras])
        else:
            out.append(item)
    return (out, metrics) if return_metrics else out


def compile_edit(clip, prompt, images_with_size=None, llama_template=None, fixed_res=None, image_positions=None, capture_spans=False, use_cache=True):
    """Encode an edit prompt with optional reference images.

    ``llama_template`` selects the encoder's chat template. When ``None``, the
    legacy ``SYS_TEMPLATE`` is used (Krea 2 default). Encoder-specific modules
    pass their own template so the same helper serves both Krea 2 and Ideogram 4.

    ``fixed_res``, when set, overrides the per-tier resolution and scales every
    image to that fixed longest-side resolution.

    ``image_positions`` is an optional list (one entry per non-None image, in
    order) of ``(x0, y0)`` merged-token-grid offsets. When set, each image's
    MRoPE h/w position ids are rewritten so the image claims to sit at that
    spot of a virtual canvas — positional placement without paying tokens for
    the canvas itself (the corner-of-black-canvas trick, token-free).

    ``capture_spans``, when True, records each image's token span
    ``(start, size)`` into the conditioning extras under ``"image_spans"`` so
    downstream nodes can blend image tokens between encodings
    (see ``QwenVLImageEmbedsMerge``).

    ``use_cache``: identical encodes are served from a module-level cache
    (keyed by clip identity, prompt, image pixel hashes, template, positions
    and flags). GPU encoding is nondeterministic, so without the cache every
    queue run produces slightly different hidden states for the same inputs.
    Cached entries are deep-copied to CPU; the caller may mutate the returned
    conditioning freely."""
    if not _COMFY_AVAILABLE:
        raise RuntimeError("Edit encode requires ComfyUI (comfy.utils, node_helpers).")

    if llama_template is None:
        llama_template = SYS_TEMPLATE

    images_vl = []
    image_prompt = ""

    if images_with_size:
        for i, (image, tier) in enumerate(images_with_size):
            if image is None:
                continue
            target = fixed_res if fixed_res is not None else RESOLUTIONS.get(tier, 256)
            samples = image.movedim(-1, 1)  # NHWC -> NCHW
            scaled = _scale_to_resolution(samples, target)
            images_vl.append(scaled.movedim(1, -1))  # back to NHWC for clip.tokenize
            image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(len(images_vl))

    full_prompt = image_prompt + prompt if image_prompt else prompt

    cache_key = None
    if use_cache:
        cache_key = (
            clip_signature(clip),
            full_prompt,
            llama_template,
            tuple(image_signature(img) for img in images_vl),
            tuple(tuple(p) if p is not None else None for p in (image_positions or [])),
            bool(capture_spans),
        )
        cached = _ENCODE_CACHE.get(cache_key)
        if cached is not None:
            return _clone_conditioning(cached)

    tokens = clip.tokenize(
        full_prompt,
        images=images_vl if images_vl else None,
        llama_template=llama_template,
    )
    pos_enter, pos_exit = _qwen_image_positions_ctx(clip, image_positions)
    span_enter, span_exit, spans = _capture_image_spans(clip) if capture_spans \
        else ((lambda: None), (lambda: None), [])
    pos_enter()
    span_enter()
    try:
        conditioning = clip.encode_from_tokens_scheduled(tokens)
    finally:
        span_exit()
        pos_exit()

    if capture_spans and spans:
        for item in (conditioning if isinstance(conditioning, list) else [conditioning]):
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[1], dict):
                item[1]["image_spans"] = [list(s) for s in spans]

    if cache_key is not None:
        _ENCODE_CACHE.put(cache_key, _clone_conditioning(conditioning, to_cpu=True))

    return conditioning


def _clone_conditioning(structure, to_cpu=False):
    """Deep-copy a conditioning structure (list of [tensor, dict] pairs).

    With ``to_cpu`` the tensors are copied to CPU so cached entries never
    pin VRAM. Mutable extras (e.g. ``image_spans`` lists) are copied too, so
    callers can mutate the returned conditioning without touching the
    cache."""
    if isinstance(structure, list):
        out = []
        for item in structure:
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                t = item[0]
                if to_cpu:
                    t = t.detach().to("cpu", copy=True)
                else:
                    t = t.clone()
                extras = dict(item[1])
                extras = {k: (list(v) if isinstance(v, list) else
                              (dict(v) if isinstance(v, dict) else v))
                          for k, v in extras.items()}
                out.append([t, extras])
            else:
                out.append(item)
        return out
    return structure


def _scale_cond_tensor(t, scale, weights=None):
    if weights is None:
        return t * scale

    flat = t.shape[-1]
    n_layers = len(weights)
    if n_layers > 1 and flat % n_layers == 0:
        layer_dim = flat // n_layers
        orig_dtype = t.dtype
        t = t.float()
        t = t.view(*t.shape[:-1], n_layers, layer_dim)
        gains = torch.tensor(weights, dtype=t.dtype, device=t.device)
        t = t * gains.view(*([1] * (t.dim() - 2)), n_layers, 1)
        t = t.view(*t.shape[:-2], flat)
        return t.to(orig_dtype) * scale
    return t * scale


def scale_conditioning(structure, scale, weights=None):
    if isinstance(structure, list):
        out = []
        for item in structure:
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                new_cond = _scale_cond_tensor(cond_t, scale, weights)
                out.append([new_cond, dict(extras)])
            else:
                out.append(scale_conditioning(item, scale, weights))
        return out
    if isinstance(structure, torch.Tensor):
        return _scale_cond_tensor(structure, scale, weights)
    if isinstance(structure, dict):
        return {k: scale_conditioning(v, scale, weights)
                for k, v in structure.items()}
    return structure


def refocus(conditioning, scale, weights):
    plw = _parse_floats(weights) if weights else None
    return scale_conditioning(conditioning, scale, weights=plw)


def _project_dissim_per_band(cond_bands, ref_bands, d, n_bands, strength, per_band_strengths, sign):
    b = cond_bands.shape[0]
    cond_mean = cond_bands.float().mean(dim=1)
    ref_mean = ref_bands.float().mean(dim=1)
    ref_mean = _match_batch(ref_mean, b)
    direction = _unit_norm_dim(cond_mean - ref_mean)

    if per_band_strengths is None:
        gains = [strength] * n_bands
    else:
        gains = list(per_band_strengths)
        if len(gains) < n_bands:
            gains = gains + [strength] * (n_bands - len(gains))
        elif len(gains) > n_bands:
            gains = gains[:n_bands]

    gains_t = torch.tensor(gains, dtype=cond_bands.float().dtype, device=cond_bands.device)
    gains_t = gains_t.view(1, 1, n_bands, 1)

    cond_f = cond_bands.float()
    dir_exp = direction.unsqueeze(1)
    proj = (cond_f * dir_exp).sum(dim=-1, keepdim=True)
    out = cond_f + sign * gains_t * proj * dir_exp
    return _merge_bands(out.to(cond_bands.dtype))


def _project_dissim_whole(cond_t, ref_t, strength, sign):
    b = cond_t.shape[0]
    cond_mean = cond_t.float().mean(dim=1, keepdim=True)
    ref_mean = ref_t.float().mean(dim=1, keepdim=True)
    ref_mean = _match_batch(ref_mean, b)
    direction = _unit_norm_dim(cond_mean - ref_mean)
    proj = (cond_t.float() * direction).sum(dim=-1, keepdim=True)
    out = cond_t.float() + sign * strength * proj * direction
    return out.to(cond_t.dtype)


def _apply_dissim(cond_t, ref_t, strength, per_band_strengths, n_bands=None):
    if n_bands is None:
        n_bands = _resolve_n_bands(cond_t)
    cond_bands, d = _split_bands(cond_t, n_bands)
    ref_bands, d2 = _split_bands(ref_t, n_bands)
    if cond_bands is not None and ref_bands is not None and d == d2:
        return _project_dissim_per_band(cond_bands, ref_bands, d, n_bands, strength, per_band_strengths, sign=+1)
    return _project_dissim_whole(cond_t, ref_t, strength, sign=+1)


def guidance_conditioning(structure, ref_structure, strength, per_band_strengths=None, n_bands=None):
    if isinstance(structure, list):
        out = []
        ref_iter = iter(ref_structure) if isinstance(ref_structure, list) else None
        for item in structure:
            ref_item = next(ref_iter, None) if ref_iter is not None else None
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                ref_t = _extract_cond_tensor(ref_item) if ref_item is not None else None
                new_cond = _apply_dissim(cond_t, ref_t, strength, per_band_strengths, n_bands=n_bands) \
                    if ref_t is not None else cond_t
                out.append([new_cond, dict(extras)])
            else:
                out.append(guidance_conditioning(item, ref_item, strength, per_band_strengths, n_bands=n_bands))
        return out
    if isinstance(structure, torch.Tensor):
        ref_t = _extract_cond_tensor(ref_structure) if ref_structure is not None else None
        if ref_t is not None:
            return _apply_dissim(structure, ref_t, strength, per_band_strengths, n_bands=n_bands)
        return structure
    return structure


def guidance(conditioning, reference, strength, n_bands=None):
    return guidance_conditioning(conditioning, reference, strength, per_band_strengths=None, n_bands=n_bands)


def _encode_mean_tokens(clip, text):
    """Encode one prompt and return its mean token vector, shape ``(d,)``."""
    tokens = clip.tokenize(text)
    cond = clip.encode_from_tokens_scheduled(tokens)
    mean = _cond_mean_tokens(cond)
    if mean is None:
        raise ValueError("empty encoding for prompt: " + text[:80])
    return mean


def _cond_mean_tokens(structure):
    """Mean token vector of a conditioning structure (list of [tensor, dict]
    items, a single such item, or a bare tensor), shape ``(d,)``, or None if
    no tensors are present."""
    if isinstance(structure, (list, tuple)):
        vecs = [v for v in (_cond_mean_tokens(item) for item in structure)
                if v is not None]
        if not vecs:
            return None
        return torch.stack(vecs).mean(dim=0)
    t = _extract_cond_tensor(structure)
    if t is None or t.numel() == 0:
        return None
    return t.detach().float().mean(dim=1).mean(dim=0)


def _parse_prompt_pairs(pairs, subjects=""):
    """Parse 'A ||| B' lines, optionally expanded over one subject per line."""
    subs = [s.strip() for s in str(subjects).splitlines()
            if s.strip() and not s.strip().startswith("#")] or [None]
    out = []
    for line in str(pairs).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|||" not in line:
            raise ValueError("each pair line needs 'A ||| B', got: " + line[:80])
        a, b = (x.strip() for x in line.split("|||", 1))
        for s in subs:
            out.append((a.replace("{subject}", s), b.replace("{subject}", s)) if s else (a, b))
    if not out:
        raise ValueError("no prompt pairs given")
    return out


def _pair_direction(clip, pairs, subjects=""):
    """Dials-style direction: mean_tokens(enc(B)) - mean_tokens(enc(A)),
    averaged over all prompt pairs (and subjects), so wording and subjects
    that differ across pairs cancel."""
    parsed = _parse_prompt_pairs(pairs, subjects)
    acc = None
    for a, b in parsed:
        d = _encode_mean_tokens(clip, b) - _encode_mean_tokens(clip, a)
        acc = d if acc is None else acc + d
    return acc / float(len(parsed))


def guidance_direction(structure, direction, strength, n_bands=None):
    """Amplify conditioning along a fixed unit direction (dial-style).

    Same projection amplification as ``guidance_conditioning``, but the
    direction is supplied by the caller (e.g. derived from prompt pairs)
    instead of being computed from a reference conditioning."""
    if isinstance(structure, list):
        return [guidance_direction(item, direction, strength, n_bands=n_bands)
                for item in structure]
    if isinstance(structure, torch.Tensor):
        d = direction.detach().to(device=structure.device)
        cond_bands, band_d = _split_bands(structure, n_bands)
        dir_bands, dir_band_d = _split_bands(d, n_bands)
        if cond_bands is not None and dir_bands is not None and band_d == dir_band_d:
            dir_unit = _unit_norm_dim(dir_bands)
            dir_exp = dir_unit.view(*([1] * (cond_bands.dim() - 2)), *dir_unit.shape)
            cond_f = cond_bands.float()
            proj = (cond_f * dir_exp).sum(dim=-1, keepdim=True)
            out = cond_f + float(strength) * proj * dir_exp
            return _merge_bands(out).to(structure.dtype)
        # whole-tensor fallback
        d_unit = _unit_norm_dim(d.reshape(1, 1, -1)).reshape(-1)
        cond_f = structure.float()
        proj = (cond_f * d_unit).sum(dim=-1, keepdim=True)
        return (cond_f + float(strength) * proj * d_unit).to(structure.dtype)
    return structure


def _top_percent_mask(t, percent, dim=-1):
    """Boolean mask selecting the top `percent` of elements by magnitude along `dim`."""
    if percent >= 1.0:
        return torch.ones_like(t, dtype=torch.bool)
    if percent <= 0.0:
        return torch.zeros_like(t, dtype=torch.bool)
    mag = t.float().abs()
    size = t.shape[dim]
    k = max(1, int(round(percent * size)))
    k = min(k, size)
    _, idx = torch.topk(mag, k, dim=dim)
    mask = torch.zeros_like(t, dtype=torch.bool)
    mask.scatter_(dim, idx, True)
    return mask


def _pad_seq(t, target_len):
    """Right-pad a conditioning tensor along the token dimension (dim 1)."""
    cur = t.shape[1]
    if cur >= target_len:
        return t
    pad_shape = list(t.shape)
    pad_shape[1] = target_len - cur
    pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=1)


def _merge_top_match_tensor(cond_a, cond_b, match_percent):
    """Equally merge two conditioning tensors.

    Elements that are in the top `match_percent` of *both* tensors by magnitude
    and share the same sign (i.e. they "match") are averaged for an equal blend.
    Elements that do not match fall back to the higher-magnitude source so no
    information is lost where the two conditionings disagree.
    """
    orig_dtype = cond_a.dtype
    a = cond_a.float()
    b = cond_b.float()
    target_batch = _common_batch([a, b])
    a = _match_batch(a, target_batch)
    b = _match_batch(b, target_batch)

    # Pad the shorter tensor to match the longer one along the token dimension.
    if a.dim() >= 2 and b.dim() >= 2 and a.shape[1] != b.shape[1]:
        target_len = max(a.shape[1], b.shape[1])
        a = _pad_seq(a, target_len)
        b = _pad_seq(b, target_len)

    mask_a = _top_percent_mask(a, match_percent)
    mask_b = _top_percent_mask(b, match_percent)
    sign_match = (a.sign() == b.sign())
    match = mask_a & mask_b & sign_match

    avg = (a + b) / 2.0
    a_dominant = a.abs() >= b.abs()
    fallback = torch.where(a_dominant, a, b)
    out = torch.where(match, avg, fallback)
    return out.to(orig_dtype)


def merge_conditioning(structure_a, structure_b, match_percent):
    if isinstance(structure_a, list):
        out = []
        b_iter = iter(structure_b) if isinstance(structure_b, list) else None
        for item in structure_a:
            b_item = next(b_iter, None) if b_iter is not None else None
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_a, extras = item
                cond_b = _extract_cond_tensor(b_item) if b_item is not None else None
                new_cond = _merge_top_match_tensor(cond_a, cond_b, match_percent) \
                    if cond_b is not None else cond_a
                out.append([new_cond, dict(extras)])
            else:
                out.append(merge_conditioning(item, b_item, match_percent))
        return out
    if isinstance(structure_a, torch.Tensor):
        cond_b = _extract_cond_tensor(structure_b) if structure_b is not None else None
        if cond_b is not None:
            return _merge_top_match_tensor(structure_a, cond_b, match_percent)
        return structure_a
    return structure_a


def _prepare_pair(cond_a, cond_b):
    """Match batch sizes and token lengths, returning float tensors.

    Returns ``(a, b, len_a, len_b)`` where the lengths are the *original*
    token counts, so callers can restore the longer input's tail after the
    merge instead of leaving it diluted by zero padding."""
    a = cond_a.float()
    b = cond_b.float()
    len_a = a.shape[1] if a.dim() >= 2 else 0
    len_b = b.shape[1] if b.dim() >= 2 else 0
    target_batch = _common_batch([a, b])
    a = _match_batch(a, target_batch)
    b = _match_batch(b, target_batch)
    if a.dim() >= 2 and b.dim() >= 2 and len_a != len_b:
        target_len = max(len_a, len_b)
        a = _pad_seq(a, target_len)
        b = _pad_seq(b, target_len)
    return a, b, len_a, len_b


def _restore_tail(out, a, b, len_a, len_b):
    """Tokens that exist only in the longer input pass through unchanged.

    Without this, the shorter input's zero padding would dilute the longer
    input's tail (e.g. an average would halve it, a min would zero it)."""
    if out.dim() < 2 or len_a == len_b:
        return out
    if len_a > len_b:
        out[:, len_b:len_a] = a[:, len_b:len_a]
    else:
        out[:, len_a:len_b] = b[:, len_a:len_b]
    return out


def _merge_average_tensor(cond_a, cond_b):
    """Plain equal average of two conditioning tensors."""
    a, b, la, lb = _prepare_pair(cond_a, cond_b)
    out = (a + b) / 2.0
    _restore_tail(out, a, b, la, lb)
    return out.to(cond_a.dtype)


def _merge_weighted_tensor(cond_a, cond_b, strength):
    """Weighted blend: ``a * (1 - strength) + b * strength``."""
    a, b, la, lb = _prepare_pair(cond_a, cond_b)
    s = float(strength)
    out = a * (1.0 - s) + b * s
    _restore_tail(out, a, b, la, lb)
    return out.to(cond_a.dtype)


def _merge_max_tensor(cond_a, cond_b):
    """Keep the element with the larger magnitude from each conditioning."""
    a, b, la, lb = _prepare_pair(cond_a, cond_b)
    a_dom = a.abs() >= b.abs()
    out = torch.where(a_dom, a, b)
    _restore_tail(out, a, b, la, lb)
    return out.to(cond_a.dtype)


def _merge_min_tensor(cond_a, cond_b):
    """Keep the element with the smaller magnitude from each conditioning."""
    a, b, la, lb = _prepare_pair(cond_a, cond_b)
    a_dom = a.abs() <= b.abs()
    out = torch.where(a_dom, a, b)
    _restore_tail(out, a, b, la, lb)
    return out.to(cond_a.dtype)


_TOKEN_METRICS = ["magnitude", "variance", "distinctiveness"]


def _token_importance(t, metric="magnitude"):
    """Score each token's informativeness, returning a ``(B, T)`` tensor.

    ``t`` is a float tensor of shape ``(B, T, D)``. Supported metrics:
      - magnitude       L1 norm across features (overall energy)
      - variance        variance across features (internal structure)
      - distinctiveness squared distance from the batch's mean token
                        (how unusual a token is vs. the typical token)
    """
    metric = (metric or "magnitude").lower()
    if metric == "variance":
        return t.var(dim=-1)
    if metric == "distinctiveness":
        mean = t.mean(dim=1, keepdim=True)
        return (t - mean).pow(2).sum(dim=-1)
    return t.abs().sum(dim=-1)


def _select_top_tokens(t, k, metric="magnitude"):
    """Keep the top-``k`` tokens of ``t`` by ``metric``, preserving order.

    Selected tokens are returned in their original sequence order (not sorted
    by importance) so positional structure is retained.
    """
    b, tt, d = t.shape
    k = int(k)
    k = max(1, min(k, tt))
    if k >= tt:
        return t
    importance = _token_importance(t, metric)  # (B, T)
    _, idx = torch.topk(importance, k, dim=-1)  # (B, k)
    idx, _ = idx.sort(dim=-1)  # preserve original token order
    idx = idx.unsqueeze(-1).expand(b, k, d)
    return t.gather(1, idx)


def _truncate_tokens(t, strength, metric="magnitude"):

    if strength >= 1.0:
        return t
    b, tt, d = t.shape
    k = max(1, int(round(strength * tt)))
    return _select_top_tokens(t, k, metric)


def _concat_with_budget(floats, target_tokens, metric="magnitude", distribute=False):

    if not all(t.dim() == 3 for t in floats):
        return torch.cat(floats, dim=1)
    target = int(target_tokens)
    if target <= 0:
        return torch.cat(floats, dim=1)
    lengths = [t.shape[1] for t in floats]
    total = sum(lengths)
    if target >= total:
        return torch.cat(floats, dim=1)

    if not distribute:
        parts = []
        remaining = target
        for t in floats:
            if remaining <= 0:
                break
            tt = t.shape[1]
            if tt <= remaining:
                parts.append(t)
                remaining -= tt
            else:
                parts.append(_select_top_tokens(t, remaining, metric))
                remaining = 0
        if not parts:
            parts = [floats[0]]
        return torch.cat(parts, dim=1)

    # Proportional split across all inputs, one token minimum each.
    n = len(floats)
    if target < n:
        return _concat_with_budget(floats, target, metric, distribute=False)
    alloc = [1] * n
    remaining = target - n
    for i in range(n):
        alloc[i] += int(round(remaining * lengths[i] / total))
    # Fix rounding drift so the total is exact.
    drift = target - sum(alloc)
    while drift != 0:
        for i in range(n):
            if drift > 0 and alloc[i] < lengths[i]:
                alloc[i] += 1
                drift -= 1
            elif drift < 0 and alloc[i] > 1:
                alloc[i] -= 1
                drift += 1
            if drift == 0:
                break
    parts = [_select_top_tokens(t, k, metric) if k < t.shape[1] else t
             for t, k in zip(floats, alloc)]
    return torch.cat(parts, dim=1)


def _merge_concat_floats(floats, strength=1.0, metric="magnitude", target_tokens=0,
                         distribute=False):

    if target_tokens:
        return _concat_with_budget(floats, target_tokens, metric, distribute)

    strength = float(strength)
    if strength <= 0.0:
        return floats[0]
    if strength < 1.0:
        if distribute:
            floats = [_truncate_tokens(t, strength, metric) if t.dim() == 3 else t
                      for t in floats]
        else:
            parts = [floats[0]]
            for t in floats[1:]:
                if t.dim() == 3:
                    t = _truncate_tokens(t, strength, metric)
                parts.append(t)
            floats = parts
    return torch.cat(floats, dim=1)


def _merge_concat_tensor(cond_a, cond_b, strength=1.0, metric="magnitude",
                         target_tokens=0, distribute=False):

    a = cond_a.float()
    b = cond_b.float()
    target_batch = _common_batch([a, b])
    a = _match_batch(a, target_batch)
    b = _match_batch(b, target_batch)
    return _merge_concat_floats([a, b], strength, metric, target_tokens,
                                distribute).to(cond_a.dtype)


def _merge_difference_tensor(cond_a, cond_b, strength):

    a, b, la, lb = _prepare_pair(cond_a, cond_b)
    out = a + (a - b) * float(strength)
    _restore_tail(out, a, b, la, lb)
    return out.to(cond_a.dtype)


def _merge_orthogonal_tensor(cond_a, cond_b, strength):

    a, b = _prepare_pair(cond_a, cond_b)
    # Per-batch-item projection of a onto b.
    bb = (b * b).sum(dim=tuple(range(1, b.dim())), keepdim=True) + 1e-8
    ab = (a * b).sum(dim=tuple(range(1, a.dim())), keepdim=True)
    proj = b * (ab / bb)
    return (a - proj * float(strength)).to(cond_a.dtype)


def _merge_tensor_mode(cond_a, cond_b, mode, match_percent=0.5, strength=0.5,
                       metric="magnitude", target_tokens=0, distribute=False):
    """Dispatch a two-tensor merge to the requested strategy."""
    mode = (mode or "top_match").lower()
    if mode == "average":
        return _merge_average_tensor(cond_a, cond_b)
    if mode == "norm_average":
        a, b, la, lb = _prepare_pair(cond_a, cond_b)
        out = (a + _match_energy(a, b)) / 2.0
        _restore_tail(out, a, b, la, lb)
        return out.to(cond_a.dtype)
    if mode == "add":
        a, b, la, lb = _prepare_pair(cond_a, cond_b)
        out = a + b
        _restore_tail(out, a, b, la, lb)
        return out.to(cond_a.dtype)
    if mode == "subtract":
        a, b, la, lb = _prepare_pair(cond_a, cond_b)
        out = a - b
        _restore_tail(out, a, b, la, lb)
        return out.to(cond_a.dtype)
    if mode == "weighted":
        return _merge_weighted_tensor(cond_a, cond_b, strength)
    if mode == "norm_weighted":
        a, b, la, lb = _prepare_pair(cond_a, cond_b)
        s = float(strength)
        b = _match_energy(a, b)
        out = a * (1.0 - s) + b * s
        _restore_tail(out, a, b, la, lb)
        return out.to(cond_a.dtype)
    if mode == "max_magnitude":
        return _merge_max_tensor(cond_a, cond_b)
    if mode == "min_magnitude":
        return _merge_min_tensor(cond_a, cond_b)
    if mode == "concat":
        return _merge_concat_tensor(cond_a, cond_b, strength, metric, target_tokens,
                                    distribute)
    if mode == "difference":
        return _merge_difference_tensor(cond_a, cond_b, strength)
    if mode == "orthogonal":
        return _merge_orthogonal_tensor(cond_a, cond_b, strength)
    return _merge_top_match_tensor(cond_a, cond_b, match_percent)


def merge_conditioning_mode(structure_a, structure_b, mode,
                            match_percent=0.5, strength=0.5,
                            metric="magnitude", target_tokens=0, distribute=False):
    """Merge two conditioning structures with an explicit strategy."""
    if isinstance(structure_a, list):
        out = []
        b_iter = iter(structure_b) if isinstance(structure_b, list) else None
        for item in structure_a:
            b_item = next(b_iter, None) if b_iter is not None else None
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_a, extras = item
                cond_b = _extract_cond_tensor(b_item) if b_item is not None else None
                new_cond = _merge_tensor_mode(cond_a, cond_b, mode, match_percent, strength,
                                              metric, target_tokens, distribute) \
                    if cond_b is not None else cond_a
                out.append([new_cond, dict(extras)])
            else:
                out.append(merge_conditioning_mode(item, b_item, mode,
                                                   match_percent, strength,
                                                   metric, target_tokens, distribute))
        return out
    if isinstance(structure_a, torch.Tensor):
        cond_b = _extract_cond_tensor(structure_b) if structure_b is not None else None
        if cond_b is not None:
            return _merge_tensor_mode(structure_a, cond_b, mode, match_percent, strength,
                                      metric, target_tokens, distribute)
        return structure_a
    return structure_a


def _merge_top_match_tensor_n(tensors, match_percent):

    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return None
    if len(tensors) == 1:
        return tensors[0]

    orig_dtype = tensors[0].dtype
    floats = [t.float() for t in tensors]
    # Match batch size to the first tensor.
    b0 = floats[0].shape[0]
    floats = [_match_batch(t, b0) for t in floats]
    # Pad the shorter tensors to match the longest along the token dimension.
    if all(t.dim() >= 2 for t in floats):
        target_len = max(t.shape[1] for t in floats)
        floats = [_pad_seq(t, target_len) for t in floats]

    stacked = torch.stack(floats, dim=0)  # (N, B, T, D)
    masks = torch.stack([_top_percent_mask(t, match_percent) for t in floats], dim=0)
    signs = torch.stack([t.sign() for t in floats], dim=0)
    # Match: in top percent of all AND all signs agree.
    match_all = masks.all(dim=0)
    sign_all = (signs == signs[0:1]).all(dim=0)
    match = match_all & sign_all

    avg = stacked.mean(dim=0)
    # Fallback: highest-magnitude tensor per element.
    mag = stacked.abs()
    dominant_idx = mag.argmax(dim=0, keepdim=False)
    fallback = stacked.gather(0, dominant_idx.unsqueeze(0)).squeeze(0)
    out = torch.where(match, avg, fallback)
    return out.to(orig_dtype)


def _prepare_many(tensors):
    """Return float tensors matched in batch size and token length."""

    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return [], []
    floats = [t.float() for t in tensors]
    lengths = [t.shape[1] if t.dim() >= 2 else 0 for t in floats]
    target_batch = _common_batch(floats)
    floats = [_match_batch(t, target_batch) for t in floats]
    if all(t.dim() >= 2 for t in floats):
        target_len = max(t.shape[1] for t in floats)
        floats = [_pad_seq(t, target_len) for t in floats]
    return floats, lengths


def _restore_tail_n(out, floats, lengths):

    if out.dim() < 2 or not lengths:
        return out
    max_len = max(lengths)
    top_two = sorted(lengths, reverse=True)[:2]
    second = top_two[1] if len(top_two) > 1 else 0
    if max_len <= second:
        return out
    idx = lengths.index(max_len)
    out[:, second:max_len] = floats[idx][:, second:max_len]
    return out


def _merge_tensors_n(tensors, mode="top_match", match_percent=0.5, strength=0.5,
                     metric="magnitude", target_tokens=0, distribute=False):

    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return None
    if len(tensors) == 1:
        return tensors[0]

    orig_dtype = tensors[0].dtype
    mode = (mode or "top_match").lower()

    if mode == "top_match":
        return _merge_top_match_tensor_n(tensors, match_percent)

    if mode == "concat":

        b0 = _common_batch(tensors)
        floats = [_match_batch(t.float(), b0) for t in tensors]
        return _merge_concat_floats(floats, strength, metric, target_tokens,
                                    distribute).to(orig_dtype)

    floats, lengths = _prepare_many(tensors)
    if not floats:
        return None

    stacked = torch.stack(floats, dim=0)  # (N, B, T, D)

    if mode == "average":
        out = stacked.mean(dim=0)
    elif mode == "add":
        out = stacked.sum(dim=0)
    elif mode == "subtract":
        out = stacked[0] - stacked[1:].sum(dim=0)
    elif mode == "max_magnitude":
        mag = stacked.abs()
        idx = mag.argmax(dim=0)
        out = stacked.gather(0, idx.unsqueeze(0)).squeeze(0)
    elif mode == "min_magnitude":
        mag = stacked.abs()
        idx = mag.argmin(dim=0)
        out = stacked.gather(0, idx.unsqueeze(0)).squeeze(0)
    elif mode == "norm_average":
        first = floats[0]
        rest = stacked[1:].mean(dim=0)
        out = (first + _match_energy(first, rest)) / 2.0
    elif mode == "weighted":
        rest = stacked[1:].mean(dim=0)
        out = stacked[0] * (1.0 - float(strength)) + rest * float(strength)
    elif mode == "norm_weighted":
        s = float(strength)
        first = floats[0]
        rest = stacked[1:].mean(dim=0)
        rest = _match_energy(first, rest)
        out = first * (1.0 - s) + rest * s
    elif mode == "difference":
        rest = stacked[1:].mean(dim=0)
        out = stacked[0] + (stacked[0] - rest) * float(strength)
    elif mode == "orthogonal":
        first = floats[0]
        rest = stacked[1:].mean(dim=0)
        bb = (rest * rest).sum(dim=tuple(range(1, rest.dim())), keepdim=True) + 1e-8
        ab = (first * rest).sum(dim=tuple(range(1, first.dim())), keepdim=True)
        proj = rest * (ab / bb)
        out = first - proj * float(strength)
    else:
        return _merge_top_match_tensor_n(tensors, match_percent)

    _restore_tail_n(out, floats, lengths)
    return out.to(orig_dtype)


def merge_conditioning_multi(structures, match_percent=0.5, mode="top_match",
                             strength=0.5, metric="magnitude", target_tokens=0,
                             distribute=False):
    """Merge a list of conditioning structures (up to N) into one."""
    structures = [s for s in structures if s is not None]
    if not structures:
        return None
    if len(structures) == 1:
        return structures[0]

    base = structures[0]
    if isinstance(base, list):
        out = []
        iters = [iter(s) if isinstance(s, list) else None for s in structures[1:]]
        for item in base:
            items = [item]
            for it in iters:
                items.append(next(it, None) if it is not None else None)
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                extras = item[1]
                cond_tensors = []
                for it_item in items:
                    t = _extract_cond_tensor(it_item) if it_item is not None else None
                    if t is not None:
                        cond_tensors.append(t)
                new_cond = _merge_tensors_n(cond_tensors, mode, match_percent, strength,
                                            metric, target_tokens, distribute) \
                    if cond_tensors else item[0]
                out.append([new_cond, dict(extras)])
            else:
                sub = [s for s in items if s is not None]
                out.append(merge_conditioning_multi(sub, match_percent, mode, strength,
                                                    metric, target_tokens, distribute))
        return out
    if isinstance(base, torch.Tensor):
        cond_tensors = []
        for s in structures:
            t = _extract_cond_tensor(s) if s is not None else None
            if t is not None:
                cond_tensors.append(t)
        if cond_tensors:
            return _merge_tensors_n(cond_tensors, mode, match_percent, strength,
                                    metric, target_tokens, distribute)
        return base
    return base


def _merge_anchor_tensor(anchor, tensors, match_percent):

    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return anchor
    if anchor is None:
        return _merge_top_match_tensor_n(tensors, match_percent)

    orig_dtype = anchor.dtype
    a = anchor.float()
    b0 = _common_batch([a] + [t.float() for t in tensors])
    a = _match_batch(a, b0)
    floats = [_match_batch(t.float(), b0) for t in tensors]
    # Pad everything to the longest token dimension.
    anchor_len = a.shape[1] if a.dim() >= 2 else 0
    input_len = max((t.shape[1] if t.dim() >= 2 else 0) for t in floats)
    if a.dim() >= 2 and all(t.dim() >= 2 for t in floats):
        target_len = max([a.shape[1]] + [t.shape[1] for t in floats])
        a = _pad_seq(a, target_len)
        floats = [_pad_seq(t, target_len) for t in floats]

    stacked = torch.stack(floats, dim=0)  # (N, B, T, D)
    anchor_mask = _top_percent_mask(a, match_percent)
    anchor_sign = a.sign()
    sign_match = torch.stack([(t.sign() == anchor_sign) for t in floats], dim=0).all(dim=0)
    match = anchor_mask & sign_match

    avg = torch.cat([a.unsqueeze(0), stacked], dim=0).mean(dim=0)
    mag = stacked.abs()
    dominant_idx = mag.argmax(dim=0, keepdim=False)
    fallback = stacked.gather(0, dominant_idx.unsqueeze(0)).squeeze(0)
    out = torch.where(match, avg, fallback)
    # Tokens that exist only in the anchor pass through unchanged — the
    # fallback only sees the (zero-padded) inputs, so without this the
    # anchor's tail would be zeroed out.
    if out.dim() >= 2 and anchor_len > input_len:
        out[:, input_len:anchor_len] = a[:, input_len:anchor_len]
    return out.to(orig_dtype)


def merge_conditioning_anchor(anchor_structure, structures, match_percent):
    """Merge a list of conditioning structures against an anchor structure."""
    structures = [s for s in structures if s is not None]
    if not structures:
        return anchor_structure

    base = anchor_structure
    if isinstance(base, list):
        out = []
        iters = [iter(s) if isinstance(s, list) else None for s in structures]
        for item in base:
            items = []
            for it in iters:
                items.append(next(it, None) if it is not None else None)
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                anchor_t, extras = item
                cond_tensors = []
                for it_item in items:
                    t = _extract_cond_tensor(it_item) if it_item is not None else None
                    if t is not None:
                        cond_tensors.append(t)
                new_cond = _merge_anchor_tensor(anchor_t, cond_tensors, match_percent) \
                    if cond_tensors else anchor_t
                out.append([new_cond, dict(extras)])
            else:
                sub = [s for s in items if s is not None]
                out.append(merge_conditioning_anchor(item, sub, match_percent))
        return out
    if isinstance(base, torch.Tensor):
        anchor_t = _extract_cond_tensor(base) if base is not None else None
        cond_tensors = []
        for s in structures:
            t = _extract_cond_tensor(s) if s is not None else None
            if t is not None:
                cond_tensors.append(t)
        if cond_tensors:
            return _merge_anchor_tensor(anchor_t, cond_tensors, match_percent)
        return base
    return base


_MERGE_MODES = [
    "top_match",
    "average",
    "norm_average",
    "weighted",
    "norm_weighted",
    "add",
    "subtract",
    "max_magnitude",
    "min_magnitude",
    "concat",
    "difference",
    "orthogonal",
]


class ConditioningMergeMulti:
    """Merge up to 5 conditionings using the requested strategy."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning_1": ("CONDITIONING",),
            "mode": (_MERGE_MODES, {"default": "top_match"}),
            "match_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
            "metric": (_TOKEN_METRICS, {"default": "magnitude"}),
            "target_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "distribute": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "conditioning_2": ("CONDITIONING",),
            "conditioning_3": ("CONDITIONING",),
            "conditioning_4": ("CONDITIONING",),
            "conditioning_5": ("CONDITIONING",),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/conditioning"

    def main(self, conditioning_1, mode="top_match", match_percent=0.5,
             strength=0.5, metric="magnitude", target_tokens=0, distribute=False,
             conditioning_2=None, conditioning_3=None,
             conditioning_4=None, conditioning_5=None):
        match_percent = float(min(max(match_percent, 0.0), 1.0))
        strength = float(min(max(strength, 0.0), 2.0))
        target_tokens = int(max(target_tokens, 0))
        structures = [conditioning_1, conditioning_2, conditioning_3,
                      conditioning_4, conditioning_5]
        structures = [s for s in structures if s is not None]
        if not structures:
            raise ValueError("ConditioningMergeMulti: at least one conditioning is required.")
        return (merge_conditioning_multi(structures, match_percent, mode, strength,
                                         metric, target_tokens, distribute),)


class ConditioningMergeList:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "mode": (_MERGE_MODES, {"default": "top_match"}),
            "match_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
            "metric": (_TOKEN_METRICS, {"default": "magnitude"}),
            "target_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "distribute": ("BOOLEAN", {"default": False}),
        }}

    INPUT_IS_LIST = True

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/conditioning"

    def main(self, conditioning, mode="top_match", match_percent=0.5, strength=0.5,
             metric="magnitude", target_tokens=0, distribute=False):
        # With INPUT_IS_LIST=True every input arrives as a list (including the
        # scalar widgets), so unwrap each to a single value.
        if isinstance(mode, (list, tuple)):
            mode = mode[0] if mode else "top_match"
        if isinstance(match_percent, (list, tuple)):
            match_percent = match_percent[0] if match_percent else 0.5
        if isinstance(strength, (list, tuple)):
            strength = strength[0] if strength else 0.5
        if isinstance(metric, (list, tuple)):
            metric = metric[0] if metric else "magnitude"
        if isinstance(target_tokens, (list, tuple)):
            target_tokens = target_tokens[0] if target_tokens else 0
        if isinstance(distribute, (list, tuple)):
            distribute = distribute[0] if distribute else False
        match_percent = float(min(max(match_percent, 0.0), 1.0))
        strength = float(min(max(strength, 0.0), 2.0))
        target_tokens = int(max(target_tokens, 0))

        # conditioning is always a list of CONDITIONING structures here.
        if not isinstance(conditioning, (list, tuple)):
            conditioning = [conditioning]
        structures = [s for s in conditioning if s is not None]
        if not structures:
            raise ValueError("ConditioningMergeList: at least one conditioning is required.")
        return (merge_conditioning_multi(structures, match_percent, mode, strength,
                                         metric, target_tokens, distribute),)


class ConditioningMerge:

    _MERGE_MODES = _MERGE_MODES

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning_1": ("CONDITIONING",),
            "conditioning_2": ("CONDITIONING",),
            "mode": (cls._MERGE_MODES, {"default": "top_match"}),
            "match_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
            "metric": (_TOKEN_METRICS, {"default": "magnitude"}),
            "target_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            "distribute": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/conditioning"

    def main(self, conditioning_1, conditioning_2, mode="top_match",
             match_percent=0.5, strength=0.5, metric="magnitude", target_tokens=0,
             distribute=False):
        match_percent = float(min(max(match_percent, 0.0), 1.0))
        strength = float(min(max(strength, 0.0), 2.0))
        target_tokens = int(max(target_tokens, 0))
        return (merge_conditioning_mode(conditioning_1, conditioning_2, mode,
                                        match_percent, strength, metric,
                                        target_tokens, distribute),)


class RebalanceGuider:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "guidance_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/conditioning"

    def main(self, positive, negative, guidance_strength=0.500):
        return (guidance(positive, negative, guidance_strength),)


class RebalanceGuiderB:
    """Rebalance Guider variant: the steering direction comes from prompt
    pairs instead of the negative conditioning.

    For each pair (A, B) the direction is mean_tokens(enc(B)) -
    mean_tokens(enc(A)); the dial is the mean over pairs, so wording and
    subjects that differ across pairs cancel. The positive conditioning is
    amplified along that direction with the same projection used by
    Rebalance Guider.

    Alternatively, connect ``conditioning_a`` and ``conditioning_b`` to use
    mean_tokens(B) - mean_tokens(A) from two conditionings directly (no clip
    or pairs needed) — a single-pair version of the same idea."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "guidance_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
        },
        "optional": {
            "clip": ("CLIP",),
            "pairs": ("STRING", {
                "multiline": True,
                "default": "a photo of a cat ||| a photo of a dog",
                "tooltip": "One pair per line: 'A ||| B'. Direction = enc(B) - enc(A), averaged over all pairs. Used when clip is connected."}),
            "subjects": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Optional subjects, one per line. Each pair is expanded for every subject so subject content cancels in the average."}),
            "conditioning_a": ("CONDITIONING", {"tooltip": "Baseline conditioning (A). When connected together with conditioning_b, replaces clip + pairs."}),
            "conditioning_b": ("CONDITIONING", {"tooltip": "Target conditioning (B). When connected together with conditioning_a, replaces clip + pairs."}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/conditioning"

    def main(self, positive, guidance_strength=0.500, clip=None, pairs="",
             subjects="", conditioning_a=None, conditioning_b=None):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Rebalance Guider B requires ComfyUI (node_helpers).")
        if conditioning_a is not None and conditioning_b is not None:
            mean_a = _cond_mean_tokens(conditioning_a)
            mean_b = _cond_mean_tokens(conditioning_b)
            if mean_a is None or mean_b is None:
                raise ValueError("conditioning_a / conditioning_b contain no tensors.")
            direction = mean_b - mean_a
        elif clip is not None:
            direction = _pair_direction(clip, pairs, subjects)
        else:
            raise ValueError("Rebalance Guider B needs either a clip (with pairs) "
                             "or both conditioning_a and conditioning_b.")
        return (guidance_direction(positive, direction, guidance_strength),)


class StepRebalance:
    """Split a conditioning schedule at a step threshold."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning_1": ("CONDITIONING",),
            "conditioning_2": ("CONDITIONING",),
            "step": ("FLOAT", {"default": 0.00, "min": 0.000, "max": 1.000, "step": 0.01}),
            "bound": ("FLOAT", {"default": 0.00, "min": 0.000, "max": 1.000, "step": 0.01}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/conditioning"

    def main(self, conditioning_1, conditioning_2, step=0.00, bound=0.00):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Step Rebalance requires ComfyUI (node_helpers).")

        step = float(min(max(step, 0.0), 1.0))
        bound = float(min(max(bound, 0.0), 1.0))

        end_1 = float(min(step + bound, 1.0))
        start_2 = float(max(step - bound, 0.0))

        cond_split_1 = node_helpers.conditioning_set_values(
            conditioning_1, {"start_percent": 0.000, "end_percent": end_1},
        )
        cond_split_2 = node_helpers.conditioning_set_values(
            conditioning_2, {"start_percent": start_2, "end_percent": 1.000},
        )
        return (cond_split_2 + cond_split_1,)


def _parse_schedule(s):

    if not s:
        return None
    s = s.replace("\n", " ").replace("\t", " ").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    points = []
    for p in parts:
        if ":" in p:
            seg, mult_s = p.rsplit(":", 1)
        else:
            seg, mult_s = p, "1.0"
        if "-" in seg:
            start_s, end_s = seg.split("-", 1)
        else:
            start_s, end_s = seg, seg
        try:
            start = float(start_s.strip())
            end = float(end_s.strip())
            mult = float(mult_s.strip())
        except ValueError:
            continue
        start = float(min(max(start, 0.0), 1.0))
        end = float(min(max(end, 0.0), 1.0))
        if end < start:
            start, end = end, start
        points.append((start, end, mult))
    if not points:
        return None
    points.sort(key=lambda x: x[0])
    return points


def _build_schedule_segments(conditioning, points, interpolation, sub_steps=8):

    if not _COMFY_AVAILABLE:
        raise RuntimeError("Rebalance CFG requires ComfyUI (node_helpers).")
    out = []
    n = len(points)
    for i, (t0, t1, m_i) in enumerate(points):
        m_next = points[i + 1][2] if i + 1 < n else m_i
        if t1 <= t0:
            t1 = max(t1, t0)
            seg = scale_conditioning(conditioning, m_i)
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
                seg = scale_conditioning(conditioning, m)
                seg = node_helpers.conditioning_set_values(
                    seg, {"start_percent": ts, "end_percent": te},
                )
                out.append(seg)
        else:
            seg = scale_conditioning(conditioning, m_i)
            seg = node_helpers.conditioning_set_values(
                seg, {"start_percent": t0, "end_percent": t1},
            )
            out.append(seg)

    combined = []
    for seg in out:
        combined = combined + seg
    return combined


class RebalanceCFG:
    """CFG-style conditioning rebalance from editable point schedule strings."""

    DEFAULT_SCHEDULE_1 = (
        "0.000-0.125:1.50; 0.125-0.250:1.40; 0.250-0.375:1.20; 0.375-0.500:1.00;"
        " 0.500-0.625:0.80; 0.625-0.750:0.60; 0.750-0.875:0.40; 0.875-1.000:0.20"
    )
    DEFAULT_SCHEDULE_2 = (
        "0.000-0.125:0.20; 0.125-0.250:0.40; 0.250-0.375:0.60; 0.375-0.500:0.80;"
        " 0.500-0.625:1.00; 0.625-0.750:1.20; 0.750-0.875:1.40; 0.875-1.000:1.50"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning_1": ("CONDITIONING",),
            "conditioning_2": ("CONDITIONING",),
            "schedule_1": ("STRING", {"default": cls.DEFAULT_SCHEDULE_1, "multiline": True}),
            "schedule_2": ("STRING", {"default": cls.DEFAULT_SCHEDULE_2, "multiline": True}),
            "interpolation": (["constant", "gradual"], {"default": "gradual"}),
            "sub_steps": ("INT", {"default": 8, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/conditioning"

    def main(self, conditioning_1, conditioning_2, schedule_1, schedule_2,
             interpolation="gradual", sub_steps=8):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Rebalance CFG requires ComfyUI (node_helpers).")

        pts1 = _parse_schedule(schedule_1)
        pts2 = _parse_schedule(schedule_2)
        if pts1 is None or pts2 is None:
            raise ValueError(
                "Rebalance CFG: invalid schedule string. "
                "Use 'start-end:multiplier; ...' (e.g. 0.000-0.125:1.5; ...)."
            )

        sub_steps = int(min(max(sub_steps, 1), 64))
        seg1 = _build_schedule_segments(conditioning_1, pts1, interpolation, sub_steps)
        seg2 = _build_schedule_segments(conditioning_2, pts2, interpolation, sub_steps)
        return (seg1 + seg2,)


class QwenVLListEncodeRebalance:
    """Encode lists of text prompts and/or images into a list of conditionings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP",),
        },
        "optional": {
            "image1": ("IMAGE",),
            "image1_tokens": (["low", "normal", "high", "max", "focal"], {"default": "normal"}),
            "image2": ("IMAGE",),
            "image2_tokens": (["low", "normal", "high", "max", "focal"], {"default": "normal"}),
            "image3": ("IMAGE",),
            "image3_tokens": (["low", "normal", "high", "max", "focal"], {"default": "normal"}),
            "image4": ("IMAGE",),
            "image4_tokens": (["low", "normal", "high", "max", "focal"], {"default": "normal"}),
            "image1_pos": ("STRING", {"default": "", "placeholder": "x,y in 0..1 of virtual canvas, empty = center"}),
            "image2_pos": ("STRING", {"default": "", "placeholder": "x,y in 0..1 of virtual canvas, empty = center"}),
            "image3_pos": ("STRING", {"default": "", "placeholder": "x,y in 0..1 of virtual canvas, empty = center"}),
            "image4_pos": ("STRING", {"default": "", "placeholder": "x,y in 0..1 of virtual canvas, empty = center"}),
        }}

    INPUT_IS_LIST = True

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/conditioning"

    @staticmethod
    def _first(value, default=None):
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value

    @staticmethod
    def _parse_pos(s):
        """Parse 'x,y' fractions (0..1) of the virtual canvas. None = default."""
        if not s:
            return None
        try:
            parts = [float(v) for v in str(s).replace(";", ",").split(",") if v.strip() != ""]
        except ValueError:
            return None
        if len(parts) != 2:
            return None
        return (min(max(parts[0], 0.0), 1.0), min(max(parts[1], 0.0), 1.0))

    @staticmethod
    def _as_items(value):
        """Normalize a slot's value into a list of images (empty if none)."""
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [v for v in value if v is not None]
        return [value]

    def main(self, text, clip,
             image1=None, image1_tokens="normal", image1_pos="",
             image2=None, image2_tokens="normal", image2_pos="",
             image3=None, image3_tokens="normal", image3_pos="",
             image4=None, image4_tokens="normal", image4_pos=""):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Qwen VL List Encode requires ComfyUI (comfy.utils, node_helpers).")

        clip = self._first(clip)
        texts = text if isinstance(text, (list, tuple)) else [text]

        slots = [
            (self._as_items(image1), self._first(image1_tokens, "normal"),
             self._parse_pos(self._first(image1_pos, ""))),
            (self._as_items(image2), self._first(image2_tokens, "normal"),
             self._parse_pos(self._first(image2_pos, ""))),
            (self._as_items(image3), self._first(image3_tokens, "normal"),
             self._parse_pos(self._first(image3_pos, ""))),
            (self._as_items(image4), self._first(image4_tokens, "normal"),
             self._parse_pos(self._first(image4_pos, ""))),
        ]

        # Number of image sets is the largest slot length (1 if no images,
        # so the node still runs once per text prompt).
        n_sets = max(len(items) for items, _, _ in slots) or 1

        results = []
        for t in texts:
            prompt = "" + _align_prompt(t)
            for i in range(n_sets):
                images_with_size = []
                positions = []
                for items, token, pos in slots:
                    img = items[min(i, len(items) - 1)] if items else None
                    images_with_size.append((img, token))
                    if img is None or pos is None:
                        positions.append(None)
                    else:
                        # Virtual canvas side in merged tokens for this tier
                        # (patch 16 * merge 2 = 32 px per grid step).
                        canvas = max(4, RESOLUTIONS.get(token, 768) // 32)
                        positions.append((int(round(pos[0] * canvas)),
                                          int(round(pos[1] * canvas))))
                has_image = any(img is not None for img, _ in images_with_size)
                images = images_with_size if has_image else None
                results.append(compile_edit(clip, prompt, images,
                                            image_positions=positions))

        return (results,)


class ConditioningFreeze:
    """Save conditioning as-is to ``ComfyUI/input/conditioning``. """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "filename": ("STRING", {"default": "conditioning.pt"}),
        }}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Rebalance-Pack/conditioning"

    def save(self, conditioning, filename):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Conditioning Freeze requires ComfyUI (folder_paths).")

        filename = str(filename).strip() or "conditioning.pt"
        if not filename.endswith(".pt"):
            filename += ".pt"

        base = folder_paths.get_input_directory()
        out_dir = os.path.join(base, "conditioning")
        os.makedirs(out_dir, exist_ok=True)

        full_path = os.path.join(out_dir, filename)
        torch.save(conditioning, full_path)

        return {"ui": {"text": ["Saved conditioning to " + full_path]}}


class ConditioningUnfreeze:
    """Load conditioning saved by Conditioning Freeze. """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "filename": ("STRING", {"default": "conditioning.pt"}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "load"
    CATEGORY = "Rebalance-Pack/conditioning"

    def load(self, filename):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Conditioning Unfreeze requires ComfyUI (folder_paths).")

        filename = str(filename).strip() or "conditioning.pt"
        if not filename.endswith(".pt"):
            filename += ".pt"

        base = folder_paths.get_input_directory()
        in_dir = os.path.join(base, "conditioning")
        full_path = os.path.join(in_dir, filename)

        if not os.path.isfile(full_path):
            raise FileNotFoundError("Conditioning file not found: " + full_path)

        conditioning = torch.load(full_path, map_location="cpu")
        return (conditioning,)


class QwenVLImageEmbedsEncode:
    """Encode text + reference images for Qwen3-VL style encoders (Krea),
    recording image token spans so encodings can be blended downstream
    (see Qwen VL Image Embeds Merge). Same encoding as Qwen VL List Encode
    Rebalance, plus span metadata.

    Supports large list batches (200+ images): with ``INPUT_IS_LIST`` every
    image input arrives as a list, and each image is encoded in its own
    forward pass (chunked by ``batch_size`` when the encoder supports batched
    image tokens) so VRAM stays bounded regardless of list length."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP",),
        },
        "optional": {
            "image1": ("IMAGE",),
            "image1_tokens": (["low", "normal", "high", "max", "focal"], {"default": "normal"}),
            "image2": ("IMAGE",),
            "image2_tokens": (["low", "normal", "high", "max", "focal"], {"default": "normal"}),
            "image3": ("IMAGE",),
            "image3_tokens": (["low", "normal", "high", "max", "focal"], {"default": "normal"}),
            "image4": ("IMAGE",),
            "image4_tokens": (["low", "normal", "high", "max", "focal"], {"default": "normal"}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/conditioning"

    @staticmethod
    def _first(value, default=None):
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value

    @staticmethod
    def _as_items(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [v for v in value if v is not None]
        return [value]

    def main(self, text, clip,
             image1=None, image1_tokens="normal",
             image2=None, image2_tokens="normal",
             image3=None, image3_tokens="normal",
             image4=None, image4_tokens="normal"):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Qwen VL Image Embeds Encode requires ComfyUI (comfy.utils).")

        clip = self._first(clip)
        texts = text if isinstance(text, (list, tuple)) else [text]

        slots = [
            (self._as_items(image1), self._first(image1_tokens, "normal")),
            (self._as_items(image2), self._first(image2_tokens, "normal")),
            (self._as_items(image3), self._first(image3_tokens, "normal")),
            (self._as_items(image4), self._first(image4_tokens, "normal")),
        ]

        # Number of encode passes = max slot length (1 if no images, so the
        # node still runs once per text prompt).
        n_sets = max(len(items) for items, _ in slots) or 1

        results = []
        for t in texts:
            prompt = "" + _align_prompt(t)
            for i in range(n_sets):
                images_with_size = []
                for items, token in slots:
                    img = items[min(i, len(items) - 1)] if items else None
                    images_with_size.append((img, token))
                has_image = any(img is not None for img, _ in images_with_size)
                images = images_with_size if has_image else None
                results.append(compile_edit(clip, prompt, images, capture_spans=True))

        return (results,)


class QwenVLImageEmbedsMerge:
    """Reduce many Qwen-VL conditionings (Krea encoder) to a single style
    conditioning. The image-token spans of all inputs are reduced with a
    batch-level strategy (see the mode list); text tokens come from the first
    input unchanged.

    Designed for large batches: with ``INPUT_IS_LIST`` it accepts a list of
    200+ conditionings and reduces them in one pass over the collected span
    tensors. ``strength`` tunes the selected strategy and is hidden for modes
    that don't use it (mean, median, dominant, rms, pca_style, concat, ...).
    Its meaning per mode:
      trimmed_mean  fraction of outliers to drop (0 = mean, 0.5 = median)
      consensus     required sign-agreement ratio (0.5 .. 1.0)
      spread        std gain on top of the mean
      weighted / norm_weighted   anchor vs crowd blend
      top_match     magnitude match_percent
      outlier       rank from most typical (0) to most extreme (1)
      random        seed (strength * 1000)"""

    _MODES = _STYLE_REDUCERS
    _STRENGTH_FREE = _STRENGTH_FREE

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "mode": (cls._MODES, {"default": "mean"}),
            "strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "style_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.05,
                                      "tooltip": "Adaptively amplify the merged style. Gain per token scales with how much the images agree along the style direction, so shared features get boosted and noise doesn't. Energy is renormalized. 0 = off; 1-2 is a good starting range; go higher (4-10+) for a hard-hitting style."}),
        },
        "optional": {
            "conditioning_2": ("CONDITIONING",),
            "harmonic_sign": (["clamp_positive", "signed"], {"default": "clamp_positive"}),
        }}

    RETURN_TYPES = ("CONDITIONING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("conditioning", "coherence", "energy", "contrast", "distinct", "coverage")
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/conditioning"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, False, False, False, False, False)

    def main(self, conditioning, mode="mean",
             strength=0.2, style_boost=0.0, conditioning_2=None,
             harmonic_sign="clamp_positive"):
        # With INPUT_IS_LIST=True widget inputs also arrive as lists; unwrap.
        if isinstance(mode, (list, tuple)):
            mode = mode[0] if mode else "mean"
        if isinstance(strength, (list, tuple)):
            strength = strength[0] if strength else 0.2
        if isinstance(style_boost, (list, tuple)):
            style_boost = style_boost[0] if style_boost else 0.0
        if isinstance(harmonic_sign, (list, tuple)):
            harmonic_sign = harmonic_sign[0] if harmonic_sign else "clamp_positive"
        mode = (mode or "mean").lower()
        harmonic_sign = (harmonic_sign or "clamp_positive").lower()
        if harmonic_sign not in ("clamp_positive", "signed"):
            harmonic_sign = "clamp_positive"
        # Strength is meaningless for strength-free modes; force a neutral
        # value so results don't depend on a stale slider.
        if mode in self._STRENGTH_FREE:
            strength = 0.0
        strength = float(min(max(strength, 0.0), 1.0))
        style_boost = float(min(max(style_boost, 0.0), 20.0))

        # Flatten both inputs into one stream of conditioning structures.
        stream = []
        for s in (conditioning, conditioning_2):
            if s is None:
                continue
            if isinstance(s, (list, tuple)):
                stream.extend([x for x in s if x is not None])
            else:
                stream.append(s)
        if not stream:
            raise ValueError("Qwen VL Image Embeds Merge: at least one conditioning is required.")

        result, metrics = reduce_image_embeds(
            None, stream, mode=mode, strength=strength,
            harmonic_sign=harmonic_sign, style_boost=style_boost,
            return_metrics=True)

        # List-mode nodes must return EVERY output wrapped in a list.
        # OUTPUT_IS_LIST only controls how downstream consumes it: the
        # conditioning slot is (True,) so [result] is consumed as a list of
        # structures; the metric slots are (False,) so ComfyUI unwraps the
        # single-element list into a plain float.
        m = metrics or {}
        return ([result],
                [m.get("coherence", 0.0)],
                [m.get("energy", 0.0)],
                [m.get("contrast", 0.0)],
                [m.get("distinct", 0.0)],
                [m.get("coverage", 0.0)])


NODE_CLASS_MAPPINGS = {
    "RebalanceGuider": RebalanceGuider,
    "RebalanceGuiderB": RebalanceGuiderB,
    "StepRebalance": StepRebalance,
    "RebalanceCFG": RebalanceCFG,
    "ConditioningMerge": ConditioningMerge,
    "ConditioningMergeMulti": ConditioningMergeMulti,
    "ConditioningMergeList": ConditioningMergeList,
    "QwenVLListEncodeRebalance": QwenVLListEncodeRebalance,
    "QwenVLImageEmbedsEncode": QwenVLImageEmbedsEncode,
    "QwenVLImageEmbedsMerge": QwenVLImageEmbedsMerge,
    "ConditioningFreeze": ConditioningFreeze,
    "ConditioningUnfreeze": ConditioningUnfreeze,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebalanceGuider": "Rebalance Guider",
    "RebalanceGuiderB": "Rebalance Guider B",
    "StepRebalance": "Step Rebalance",
    "RebalanceCFG": "Rebalance CFG Custom",
    "ConditioningMerge": "Conditioning Merge",
    "ConditioningMergeMulti": "Conditioning Merge (Multi)",
    "ConditioningMergeList": "Conditioning Merge (List)",
    "QwenVLListEncodeRebalance": "Qwen VL List Encode Rebalance",
    "QwenVLImageEmbedsEncode": "Qwen VL Image Embeds Encode",
    "QwenVLImageEmbedsMerge": "Qwen VL Image Embeds Merge",
    "ConditioningFreeze": "Conditioning Freeze",
    "ConditioningUnfreeze": "Conditioning Unfreeze",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",


    # Core helpers re-exported
    "EncoderProfile",
    "register_encoder_profile",
    "get_encoder_profile",
    "detect_encoder_profile",
    "compile_edit",
    "scale_conditioning",
    "refocus",
    "guidance",
    "guidance_conditioning",
    "merge_conditioning",
    "merge_conditioning_multi",
    "merge_conditioning_mode",
    "merge_image_embeds",
    "ConditioningMerge",
    "ConditioningMergeMulti",
    "ConditioningMergeList",
    "RebalanceCFG",
    "RebalanceGuider",
    "StepRebalance",
    "QwenVLListEncodeRebalance",
    "QwenVLImageEmbedsEncode",
    "QwenVLImageEmbedsMerge",
    "SYS_TEMPLATE",
    "RESOLUTIONS",
]
