# 24/11 Latest Update
* New sampling nodes
* Block-weighted LoRA loaders for MOSS-Image and MiniMax H3 / H3 Image
* Omni Node moved to another repository. [OmniNode](https://github.com/nova452/OmniNode)

# 10/08
* New core utility nodes
* The Omni Node family now lives in its own pack, ComfyUI-Omni-Node

# 12/07
* Supports multi character references for stronger in-context posing
* New utility nodes added

<img width="2838" height="1387" alt="Screenshot (586)" src="https://github.com/user-attachments/assets/fbc29244-021b-4c7e-90f0-f0138342e15d" />

[workflow.json](https://github.com/nova452/Rebalance-Pack/blob/main/workflows/Krea%202%20Edit%20(Full%20Context).json)

# 29/06
* Clear logic allowing for more complex edits
* Now supports negative prompting for stripping undesired effects

<img width="1623" height="1227" alt="Screenshot (563)" src="https://github.com/user-attachments/assets/35ad346b-b4f7-4a83-8181-2891357d5dd0" />

[workflow.json](https://github.com/nova452/Rebalance-Pack/blob/main/workflows/krea2_image_edit_workflow%20(3).json)

# Image Editing with Krea 2 and Ideogram 4 Models
<img width="2616" height="2463" alt="showcase (1)" src="https://github.com/user-attachments/assets/aae61b58-a0b2-49c3-8718-ec32b0ec180e" />

# Rebalance Pack
A set of evolving nodes that streamline the best use practices in ComfyUI.


foundational: Input, Any, Switch, Load Images, Load Images (Blaze), Load Image Newest, Load Image Full, Save Images, Color Image, Concatenate, String (Inline), String (Inline 5), String to List, List String Index, List Display, Float Exact, Float 1.00, Float 10.00

utility: Image Resolution Cap, Image Aspect Ratio Crop, Mask Aspect Ratio BBox, Uncrop Image, Uncrop Mask, Border Mask & Crop Detector, Mask Opacity

conditioning: Conditioning Merge, Conditioning Merge (Multi), Conditioning Merge (List), Conditioning Freeze, Conditioning Unfreeze, Qwen VL List Encode Rebalance, Qwen VL Image Embeds Encode, Qwen VL Image Embeds Merge

sampling: Rebalance Guider, Rebalance Guider B, Rebalance CFG Custom, Step Rebalance, Sigmas Rebalance, Sampler Rebalance

lora: Load LoRA (Block), Load LoRA Krea2 (Block), Load LoRA Moss Image (Block), Load LoRA H3 (Block), Load LoRA H3 Image (Block), Load LoRA Folder (Block), Load LoRA Folder Moss Image (Block), Load LoRA Folder H3 (Block), Load LoRA Folder H3 Image (Block)

model specific: Conditioning Krea2 Rebalance, Krea 2 Image Edit Rebalance, Krea 2 Image Edit Rebalance Steering, Krea 2 Encode Rebalance, Conditioning Ideogram4 Rebalance, Ideogram 4 Image Edit Rebalance, Ideogram 4 Encode Rebalance

### Sigmas Rebalance
Step Rebalance for a sigma schedule. Splits `sigmas` into `sigmas_high` and `sigmas_low` at
schedule position `step` (0 = first sigma, 1 = last, fractional so the split can fall inside a
single step). A **negative** `bound` cuts a gap between them, so `sigmas_low` starts at a lower
sigma than `sigmas_high` ended on; a positive `bound` makes them overlap. Run `sigmas_high` in one
SamplerCustomAdvanced, then `sigmas_low` in a second one with noise disabled (Disable Noise) on
the first sampler's output.

`bound` is measured by **impact**, the change in signal-to-noise the gap causes, so the same value
has the same strength at any `step` and any step count: −0.01 is as strong as a 0.02 sigma gap at
sigma 0.5. (Measured in plain sigma, the same gap is mild mid-schedule and very strong near sigma 1
or 0: a gap from a down to b changes signal-to-noise by ln((1 − b) / b) − ln((1 − a) / a) on flow
models, so 0.004 at sigma 0.995 is about as strong as 0.14 at sigma 0.5. EPS / v models use
log-sigma.)

`steps` 0 uses the incoming sigmas as they are. At 1 or more the schedule is recalculated first,
keeping the incoming first and last sigma: `resample` stretches the incoming curve to the new step
count (smoothly, keeping a closing step to 0 as its own step; it needs a few incoming steps to know
the curve, a 1-step input is a straight line); any ComfyUI scheduler rebuilds the steps between
those two sigmas with its own spacing and the connected `model`'s own sampling, like `denoise` on
BasicScheduler (any model works). Without a model it falls back to a flow schedule with `shift`
(default 1.0; MOSS-Image is 5.0). So the low half of an 8-step schedule (0.79 → 0) with `steps` 20
and `normal` becomes 20 normal-spaced steps from 0.79 to 0; a full 1.0 → 0 input gives exactly what
BasicScheduler would at that step count.

### Sampler Rebalance
The extra noise a Sigmas Rebalance cut leaves behind, added at every step without cutting the
schedule, inside one SamplerCustomAdvanced. Put it between KSamplerSelect and the sampler input.
Between steps the latent is split into the clean image the model just predicted and the noise
around it, and that noise is turned up so the signal-to-noise drops by the cut's impact. It is the
latent's own noise made stronger, not new random noise, and the sigmas stay exactly as given.
`bound` is the total impact, on the same scale as Sigmas Rebalance: −0.01 leaves as much extra noise
in total as one Sigmas Rebalance cut at −0.01 (noise ×1.083 on flow models), split evenly over the
steps between schedule positions `start` and `end`, whatever the step count. Positive values turn
the noise down instead. `detail_size` (latent pixels, default 1.5; MOSS / Flux: 1 latent pixel =
16 image pixels) boosts only noise finer than that, so the large-scale noise the model reads as
layout in the early steps is left alone and early boosts don't double heads or bodies. 0 boosts
all of the noise. `bound` 0 is the plain sampler. Multistep samplers (dpmpp_2m,
res_multistep…) lose their history across steps and act first-order; dpm_fast / dpm_adaptive are
not supported.

### Load LoRA Folder (Block) / Moss Image / H3 / H3 Image
Bulk block adjuster. Point it at a loras subfolder - every LoRA in it loads automatically, and
the node provides one 0.0-1.0 widget per block group, **defaulting to 0.0**. Each widget scales
that block across the entire collection, so a 40-LoRA stack only fires the blocks you turn up
(useful when deep blocks otherwise get fried). The Krea 2 variant uses the ModelMergeKrea2 layout
(`blocks.0-27`, `txtfusion` blocks, embedders, `last`, `other`); the Moss Image variant uses the
native MOSSImage module layout (`layers.0-41`, `text_embedder`, embedders, final stage, `other`),
and the H3 / H3 Image variants use the native MiniMax H3 layout (H3 Image drops the audio modules).
The `method` dropdown chooses the folder merge: stack (add), average, max/min magnitude per module,
or strongest file only. `strength` is a global multiplier; every run writes a summary to `report`.

### Load LoRA Krea2 (Block)
Krea 2 LoRA loader with per-block weights. Pick a LoRA, press **Scan Blocks**, and the node shows
one 0.0-1.0 widget for every block group found in the file (same layout as ModelMergeKrea2:
`first`, `tmlp`, `txtmlp`, `tproj`, `txtfusion.layerwise_blocks.0-1`, `txtfusion.projector`,
`txtfusion.refiner_blocks.0-1`, `blocks.0-27`, `last`). Groups the LoRA doesn't touch are hidden;
each block weight is a multiplier on top of `strength`.

### Load LoRA Moss Image (Block)
MOSS-Image counterpart of the Krea 2 loader. Pick a LoRA, press **Scan Blocks**, and the node shows
one 0.0-1.0 widget for every block group found in the file (native module layout of
`comfy.ldm.moss_image.model.MOSSImage`: `video_embedder`, `text_embedder.proj`,
`text_embedder.blocks.0-1`, `time_embedding`, `shared_block_modulation`, `layers.0-41`,
`final_norm_video`, `final_linear_video`, `video_head_modulation`). Groups the LoRA doesn't touch
are hidden; each block weight is a multiplier on top of `strength`.

### Load LoRA H3 (Block) / Load LoRA H3 Image (Block)
MiniMax H3 counterparts of the Krea 2 loader. Pick a LoRA, press **Scan Blocks**, and the node
shows one 0.0-1.0 widget for every block group found in the file (native module layout of
`comfy.ldm.minimax.model.MiniMaxH3Model`: patch projections, `condition_proj`, `time_embedder`,
`token_refiner.blocks.0-1`, and the 50 main blocks). H3 Image is the same model without the audio
stream, so its list drops the audio modules. Groups the LoRA doesn't touch are hidden; each block
weight is a multiplier on top of `strength`.

```bash
git clone https://github.com/nova452/Rebalance-Pack.git
```
