import torch

try:
    import comfy.samplers
    import comfy.model_sampling
    _SCHEDULERS = list(comfy.samplers.SCHEDULER_NAMES)
    _COMFY_AVAILABLE = True
except ImportError:
    _SCHEDULERS = []
    _COMFY_AVAILABLE = False


def _flow_model_sampling(shift):
    if not _COMFY_AVAILABLE:
        raise RuntimeError("Sigmas Rebalance needs ComfyUI to recalculate sigmas with a scheduler.")

    class _Config:
        sampling_settings = {"shift": float(shift), "multiplier": 1.0}

    return comfy.model_sampling.ModelSamplingDiscreteFlow(_Config())


def _position(sigmas, value):
    """Fractional position (0 = first, 1 = last) where a decreasing schedule passes value."""
    n = sigmas.shape[0] - 1
    if value >= sigmas[0]:
        return 0.0
    if value <= sigmas[-1]:
        return 1.0
    i = int((sigmas > value).sum()) - 1  # sigmas[i] > value >= sigmas[i + 1]
    span = float(sigmas[i] - sigmas[i + 1])
    f = float(sigmas[i] - value) / span if span > 0 else 0.0
    return (i + f) / n


def _flow_sigmas(scheduler, steps, shift, high, low, model_sampling=None):
    """steps sigmas from high down to low, spaced like this scheduler (and model or shift).

    Only the two end values are used, so the result does not depend on how many steps the
    incoming sigmas had. A range that is the scheduler's whole schedule gives exactly what
    BasicScheduler gives. A range that runs down to the end is done like denoise on
    BasicScheduler: the scheduler at a larger step count, its last steps kept, so the closing
    step is the scheduler's own. Any other range is read off a dense run of the scheduler.
    The result is nudged so it starts and ends exactly on high and low.
    """
    out = torch.linspace(high, low, steps + 1, dtype=torch.float64)
    if high - low <= 1e-7:
        return out.float()  # nothing to schedule: a flat or reversed range
    if model_sampling is None:
        model_sampling = _flow_model_sampling(shift)
    calc = lambda n: comfy.samplers.calculate_sigmas(model_sampling, scheduler, int(n)).double().cpu().flatten()

    own = calc(steps)
    if own.shape[0] == steps + 1 and abs(float(own[0]) - high) < 1e-5 and abs(float(own[-1]) - low) < 1e-5:
        return own.float()

    dense = calc(_DENSE)
    end = float(dense[-1])
    p_high, p_low = _position(dense, high), _position(dense, low)
    run = None
    if low <= end + 1e-7 and p_high < 1.0:
        # denoise-style: the last steps of a longer schedule, so the tail is the scheduler's own
        guess = steps / max(1.0 - p_high, 1e-9)
        if guess <= _MAX_TOTAL:
            best = None
            for total in {max(steps, int(guess) + d) for d in (-1, 0, 1, 2)}:
                s = calc(total)
                if s.shape[0] < steps + 1:
                    continue
                s = s[-(steps + 1):]
                err = abs(float(s[0]) - high)
                if best is None or err < best[0]:
                    best = (err, s)
            if best is not None:
                run = best[1]
    if run is None:
        n = dense.shape[0] - 1
        run = _pchip(dense, torch.linspace(p_high * n, p_low * n, steps + 1, dtype=torch.float64))

    top, bottom = float(run[0]), float(run[-1])
    if top - bottom > 1e-12:
        run = low + (run - bottom) * ((high - low) / (top - bottom))
    else:
        run = out
    run[0], run[-1] = high, low
    return run.float()


_DENSE = 4096       # resolution of the reference run used to read off arbitrary ranges
_MAX_TOTAL = 4000   # largest schedule built for a denoise-style range; past it the dense run is read instead


def _sigma_at(sigmas, pos):
    """Sigma at a fractional schedule position (0 = first sigma, 1 = last), linear between steps."""
    n = sigmas.shape[0] - 1
    x = pos * n
    i = min(int(x), n - 1)
    f = x - i
    return sigmas[i] + (sigmas[i + 1] - sigmas[i]) * f, i, f


def _head(sigmas, pos, eps):
    """Sigmas from the start up to position pos, ending on the interpolated sigma there."""
    if pos >= 1.0:
        return sigmas.clone()
    s, i, f = _sigma_at(sigmas, pos)
    head = sigmas[:i + 1]
    if f > 0.0 and (head[-1] - s).abs() > eps:
        head = torch.cat([head, s.reshape(1)])
    return head


def _tail(sigmas, pos, eps):
    """Sigmas from position pos to the end, starting on the interpolated sigma there."""
    if pos <= 0.0:
        return sigmas.clone()
    s, i, f = _sigma_at(sigmas, pos)
    tail = sigmas[i + 1:]
    if f == 0.0 or (tail[0] - s).abs() > eps:
        tail = torch.cat([s.reshape(1), tail])
    return tail


def _pchip(y, xq):
    """Monotone cubic (Fritsch-Carlson) through y at positions 0..n-1, read at xq. No overshoot."""
    n = y.shape[0]
    if n < 3:
        i = xq.floor().clamp(0, n - 2).long()
        f = xq - i
        return y[i] + (y[i + 1] - y[i]) * f
    d = y[1:] - y[:-1]
    m = torch.zeros_like(y)
    same = d[:-1] * d[1:] > 0
    m[1:-1] = torch.where(same, 2 * d[:-1] * d[1:] / (d[:-1] + d[1:]).where(same, torch.ones_like(d[1:])), torch.zeros_like(d[1:]))
    m[0], m[-1] = d[0], d[-1]
    i = xq.floor().clamp(0, n - 2).long()
    f = xq - i
    f2, f3 = f * f, f * f * f
    return ((2 * f3 - 3 * f2 + 1) * y[i] + (f3 - 2 * f2 + f) * m[i]
            + (-2 * f3 + 3 * f2) * y[i + 1] + (f3 - f2) * m[i + 1])


def _resample(sigmas, steps):
    """The same schedule curve at a new step count: steps + 1 sigmas evenly spaced in position.

    The first and last sigma are kept exactly; the ones in between are read off the incoming
    curve with a smooth monotone interpolation, so its scheduler and shift carry over without
    having to be known. A closing step down to 0 that is smaller than the step before it is the
    scheduler's own tail (normal appends it after its last real sigma), so it is set aside, the curve is stretched to steps - 1, and
    the 0 is put back: the tail step stays the size the scheduler gave it relative to the curve.
    """
    s = sigmas.double()
    # a separate tail (normal, karras...): the last sigma before 0 is smaller than the step leading to it;
    # simple-style schedules just carry on to 0 and are stretched as one curve
    tail = (steps >= 2 and s.shape[0] >= 3 and float(s[-1]) == 0.0
            and float(s[-2]) < float(s[-3] - s[-2]))
    body = s[:-1] if tail else s
    k = steps - 1 if tail else steps
    xq = torch.arange(k + 1, dtype=torch.float64) * ((body.shape[0] - 1) / k)
    out = _pchip(body, xq)
    out[0], out[-1] = body[0], body[-1]
    if tail:
        out = torch.cat([out, s.new_zeros(1)])
    return out.to(sigmas.dtype)


import math


def _is_flow(sigmas, model_sampling=None):
    """Flow models run sigma 1 -> 0; EPS / v models have sigmas well above 1."""
    top = getattr(model_sampling, "sigma_max", None) if model_sampling is not None else None
    top = float(top) if top is not None else float(sigmas.max())
    return top <= 1.0 + 1e-6


def _lam(s, flow):
    """Log signal-to-noise of a sigma: ln((1 - s) / s) on flow models, -2 ln s otherwise."""
    if flow:
        s = min(max(float(s), 1e-9), 1.0 - 1e-9)
        return math.log((1.0 - s) / s)
    return -2.0 * math.log(max(float(s), 1e-12))


def _sig(lam, flow):
    """Inverse of _lam."""
    lam = min(max(lam, -700.0), 700.0)
    return 1.0 / (1.0 + math.exp(lam)) if flow else math.exp(-lam / 2.0)


def _impact(bound):
    """Total signal-to-noise jump of an impact bound: what a sigma gap of 2 * |bound| centred on
    sigma 0.5 does on a flow model, so a value means the same strength anywhere on the curve."""
    b = min(abs(bound), 0.4999)
    return 2.0 * math.log((0.5 + b) / (0.5 - b))


class SigmasRebalance:
    """Split a sigma schedule at a step threshold, Step Rebalance style.

    The split sits at schedule position step (fractional, so it can fall inside a single step).
    bound is measured by impact, the signal-to-noise change of the gap: the same bound has the
    same strength wherever step is and whatever the step count, sized like a sigma gap of
    2 * |bound| at sigma 0.5. A negative bound leaves a gap: sigmas_low starts at a lower sigma
    than sigmas_high ended on, so the second sampler is handed a latent that still carries the
    noise of the cut region. A positive bound makes the halves overlap by the same amount.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sigmas": ("SIGMAS",),
            "step": ("FLOAT", {"default": 0.500, "min": 0.000, "max": 1.000, "step": 0.001, "round": 0.001, "precision": 3,
                               "tooltip": "Where the split sits in the schedule: 0 = first sigma, 1 = last sigma."}),
            "bound": ("FLOAT", {"default": 0.0000, "min": -0.4999, "max": 0.4999, "step": 0.0001, "round": 0.0001, "precision": 4,
                                "tooltip": "Size of the cut by its impact (signal-to-noise change), so the same value "
                                           "has the same strength anywhere and at any step count: -0.01 is as strong as "
                                           "a 0.02 sigma gap at sigma 0.5. Negative: a gap between the halves. "
                                           "Positive: the halves overlap."}),
            "steps": ("INT", {"default": 0, "min": 0, "max": 10000,
                              "tooltip": "0 = use the incoming sigmas as they are. 1 or more = recalculate the schedule "
                                         "at this many steps before splitting."}),
            "scheduler": (["resample"] + _SCHEDULERS, {"default": "resample",
                          "tooltip": "Used when steps is 1 or more. Both keep the incoming first and last sigma. "
                                     "resample: stretch the incoming curve to the new step count (shift is ignored). "
                                     "Any other: rebuild the steps between those two sigmas with that scheduler, using the "
                                     "connected model (or shift when no model is connected)."}),
            "shift": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01,
                                "tooltip": "Only used by a rebuilt schedule when no model is connected: flow shift "
                                           "(1.0 = none). With a model connected, its own sampling settings are used."}),
        }, "optional": {
            "model": ("MODEL", {"tooltip": "Optional. Rebuilt schedules use this model's own sampling (shift, "
                                           "sigma range), exactly like BasicScheduler. Works for any model."}),
        }}

    RETURN_TYPES = ("SIGMAS", "SIGMAS")
    RETURN_NAMES = ("sigmas_high", "sigmas_low")
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/sampling"

    def main(self, sigmas, step=0.50, bound=0.00, steps=0, scheduler="resample", shift=1.0, model=None):
        sigmas = sigmas.flatten()
        ms = model.get_model_object("model_sampling") if model is not None else None
        if steps > 0 and scheduler != "resample" and sigmas.shape[0] >= 2:
            sigmas = _flow_sigmas(scheduler, int(steps), shift, float(sigmas[0]), float(sigmas[-1]), ms).to(sigmas)
        if sigmas.shape[0] < 2:
            return (sigmas.clone(), sigmas.clone())
        if steps > 0 and scheduler == "resample":
            sigmas = _resample(sigmas, int(steps))

        step = float(min(max(step, 0.0), 1.0))
        end_1 = start_2 = step
        if bound != 0.0:
            # the gap is placed in signal-to-noise around the sigma at step, then read back as positions
            flow = _is_flow(sigmas, ms)
            half = _impact(bound) / 2.0
            centre = _lam(_sigma_at(sigmas, step)[0], flow)
            earlier, later = _sig(centre - half, flow), _sig(centre + half, flow)  # higher, lower sigma
            if bound < 0:
                end_1, start_2 = _position(sigmas, earlier), _position(sigmas, later)
            else:
                end_1, start_2 = _position(sigmas, later), _position(sigmas, earlier)

        eps = 1e-6
        return (_head(sigmas, end_1, eps), _tail(sigmas, start_2, eps))


def _find_model_sampling(model):
    """The model_sampling behind the denoiser a k-sampler is handed (KSamplerX0Inpaint -> guider -> model)."""
    m = model
    for _ in range(6):
        if m is None:
            return None
        if hasattr(m, "model_sampling"):
            return m.model_sampling
        m = getattr(m, "inner_model", None)
    return None


def _blur(x, size):
    """Gaussian blur over the last two (spatial) dims, sigma = size latent pixels."""
    r = max(1, int(math.ceil(3.0 * size)))
    k = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-(k * k) / (2.0 * size * size))
    k = (k / k.sum()).to(x.dtype)
    h, w = x.shape[-2], x.shape[-1]
    y = x.reshape(-1, 1, h, w)
    mode = "reflect" if r < h and r < w else "replicate"
    y = torch.nn.functional.conv2d(torch.nn.functional.pad(y, (r, r, 0, 0), mode=mode), k.view(1, 1, 1, -1))
    y = torch.nn.functional.conv2d(torch.nn.functional.pad(y, (0, 0, r, r), mode=mode), k.view(1, 1, -1, 1))
    return y.reshape(x.shape)


class _PerStepNoise:
    """Runs the wrapped sampler one step at a time and, between steps, boosts the noise the latent
    already carries, with the schedule left as it is.

    What a cut does: the latent goes on to the next step with more noise in it than its label says,
    its signal-to-noise lowered by the cut's impact. Here the same mismatch is made without moving
    any sigma: the latent is split into the clean image the model just predicted and the noise
    around it, and that noise is scaled up so the signal-to-noise drops by the same amount (flow:
    noise x e^impact; EPS / v: x e^(impact / 2)). It is the latent's own noise, made stronger, not
    new random noise, so the result is deterministic. bound is the total impact, on the Sigmas
    Rebalance scale, split evenly over the steps between start and end.

    With detail_size > 0 only the fine part of that noise is boosted: the noise minus a Gaussian
    blur of it (sigma = detail_size latent pixels). The large-scale part, which the model reads as
    layout in the early steps, is left exactly as the sampler made it.
    """

    def __init__(self, inner, bound, start, end, detail_size=0.0):
        self.inner = inner
        self.detail_size = detail_size
        self.bound = bound
        self.start = start
        self.end = end
        self.__name__ = getattr(inner, "__name__", "sampler") + "_rebalance"

    def __call__(self, model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        if self.bound == 0.0:  # exactly the wrapped sampler, same noise and multistep history
            return self.inner(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, **kwargs)
        extra_args = {} if extra_args is None else extra_args
        n = len(sigmas) - 1
        seed = extra_args.get("seed")
        flow = _is_flow(sigmas, _find_model_sampling(model))
        boost = [i for i in range(1, n) if self.start <= i / n <= self.end and sigmas[i] > 0]
        per_step = _impact(self.bound) / len(boost) if boost else 0.0
        if self.bound > 0:
            per_step = -per_step  # positive bound: the opposite, the latent's noise is turned down
        gain = math.exp(per_step if flow else per_step / 2.0)
        boost = set(boost)
        last = {}

        def step_cb(d, i):
            last["denoised"] = d["denoised"]
            if callback is not None:
                callback({**d, "i": i})

        for i in range(n):
            if i in boost and "denoised" in last:
                s = sigmas[i]
                den = last["denoised"]
                # the latent's noise part, around the model's last clean prediction
                noise_part = x - (1.0 - s) * den if flow else x - den
                if self.detail_size > 0.0 and x.ndim >= 3:
                    noise_part = noise_part - _blur(noise_part, self.detail_size)  # fine part only
                x = x + (gain - 1.0) * noise_part
            args = dict(extra_args)
            if seed is not None:
                args["seed"] = seed + i  # fresh noise per step for ancestral / SDE samplers
            x = self.inner(model, x, sigmas[i:i + 2], extra_args=args,
                           callback=lambda d, i=i: step_cb(d, i), disable=True, **kwargs)
        return x


class SamplerRebalance:
    """The noise a Sigmas Rebalance cut leaves behind, added at every step without any cut."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sampler": ("SAMPLER",),
            "bound": ("FLOAT", {"default": 0.0000, "min": -0.4999, "max": 0.4999, "step": 0.0001, "round": 0.0001, "precision": 4,
                                "tooltip": "Total impact, the same scale as Sigmas Rebalance: -0.01 leaves as much extra "
                                           "noise in total as one Sigmas Rebalance cut at -0.01, but without cutting the "
                                           "schedule, split evenly over the steps between start and end. Negative: more "
                                           "noise. Positive: less. 0 = the sampler unchanged."}),
            "start": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001, "round": 0.001, "precision": 3,
                                "tooltip": "First schedule position (0 = first sigma, 1 = last) where noise is boosted."}),
            "end": ("FLOAT", {"default": 1.000, "min": 0.000, "max": 1.000, "step": 0.001, "round": 0.001, "precision": 3,
                              "tooltip": "Last schedule position where noise is boosted."}),
            "detail_size": ("FLOAT", {"default": 1.50, "min": 0.00, "max": 16.00, "step": 0.05,
                                      "tooltip": "Only noise finer than this many latent pixels is boosted (MOSS / Flux: "
                                                 "1 latent pixel = 16 image pixels, SD: 8), so layout, pose and subject "
                                                 "count are left alone. Bigger = coarser detail is boosted too. "
                                                 "0 = boost all of the noise."}),
        }}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/sampling"

    def main(self, sampler, bound=0.000, start=0.0, end=1.0, detail_size=1.5):
        if not hasattr(sampler, "sampler_function"):
            raise ValueError("Sampler Rebalance needs a regular k-sampler (KSamplerSelect and similar).")
        name = getattr(sampler.sampler_function, "__name__", "")
        if name in ("dpm_fast_function", "dpm_adaptive_function"):
            raise ValueError("Sampler Rebalance can't run dpm_fast / dpm_adaptive one step at a time.")
        wrapped = _PerStepNoise(sampler.sampler_function, float(bound), float(start), float(end), float(detail_size))
        return (comfy.samplers.KSAMPLER(wrapped, dict(sampler.extra_options), dict(sampler.inpaint_options)),)


NODE_CLASS_MAPPINGS = {
    "SigmasRebalance": SigmasRebalance,
    "SamplerRebalance": SamplerRebalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SigmasRebalance": "Sigmas Rebalance",
    "SamplerRebalance": "Sampler Rebalance",
}
