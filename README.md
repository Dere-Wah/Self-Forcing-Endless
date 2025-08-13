<p align="center">
<h1 align="center">Self Forcing</h1>
<h3 align="center">Bridging the Train-Test Gap in Autoregressive Video Diffusion</h3>
</p>
<p align="center">
  <p align="center">
    <a href="https://www.xunhuang.me/">Xun Huang</a><sup>1</sup>
    ·
    <a href="https://zhengqili.github.io/">Zhengqi Li</a><sup>1</sup>
    ·
    <a href="https://guandehe.github.io/">Guande He</a><sup>2</sup>
    ·
    <a href="https://mingyuanzhou.github.io/">Mingyuan Zhou</a><sup>2</sup>
    ·
    <a href="https://research.adobe.com/person/eli-shechtman/">Eli Shechtman</a><sup>1</sup><br>
    <sup>1</sup>Adobe Research <sup>2</sup>UT Austin
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2506.08009">Paper</a> | <a href="https://self-forcing.github.io">Website</a> | <a href="https://huggingface.co/gdhe17/Self-Forcing/tree/main">Models (HuggingFace)</a></h3>
</p>

<p align="center"><i>
This repository extends the paper's implementation with new demo features (endless generation, synchronized cache purge strength, and a generator class refactor). For a practical overview and motivation, see my blog post:
<a href="https://derewah.dev/projects/self-forcing-endless">Self-Forcing: Making AI Video Generation Endless</a>.
</i></p>

---

Self Forcing trains autoregressive video diffusion models by **simulating the
inference process during training**, performing autoregressive rollout with KV
caching. It resolves the train-test distribution mismatch and enables
**real-time, streaming video generation on a single RTX 4090** while matching
the quality of state-of-the-art diffusion models.

---

https://github.com/user-attachments/assets/0392bf33-db45-49ca-b875-b773f8c19480

## Requirements

We tested this repo on the following setup:

-   Nvidia GPU with at least 24 GB memory (RTX 4090, A100, and H100 are tested).
-   Linux operating system.
-   64 GB RAM.

Other hardware setup could also work but hasn't been tested.

## Installation

Create a conda environment and install dependencies:

```
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Quick Start

### Download checkpoints

```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### GUI demo

```
python server.py
```

## New Additions

I added several features and refactors to make the demo more powerful and
interactive, inspired by real-time experimentation and explained in detail in
the blog post:
[Self-Forcing: Making AI Video Generation Endless](https://derewah.dev/projects/self-forcing-endless).

### Endless Generation

-   **What**: Continuous, theoretically unbounded video generation. The model
    now rolls forward block-by-block using a noise buffer and a rolling KV
    cache, rather than stopping after a fixed number of frames (the blog post
    discusses the historical 81-frame limit and how to go beyond it).
-   **How**: The generator advances using
    a moving start index while maintaining an internal KV cache window. Decoding
    happens per block, and frames are streamed out as they are produced.
-   **Where**: Implemented in
    `SelfForcingEndlessGenerator.endless_generation_task` inside
    `self_forcing.py`.

### Cache Purge with Strength Control

-   **Strength parameter**: `purge_strength` is an integer in `[0, 3]` that
    controls when in the denoising cycle the purge is applied (earlier =
    stronger reset).
    -   `strength 3` = earliest step (step 0) (strongest change, most reactive)
    -   `strength 2` = early-mid (step 1)
    -   `strength 1` = late-mid (step 2)
    -   `strength 0` = latest step (step 3) (subtle change, most continuity)
-   **Where**: Use `SelfForcingEndlessGenerator.purge_cache(purge_strength)` or
    the Socket.IO event `purge_cache` with a `purge_strength` payload.
    Internally, see `_denoise_frame` in `self_forcing.py`.

### Flicker Reduction via VAE Memory Handling

-   **What**: During continuous generation, the VAE’s decoding cache is maintained
    to preserve consistency across blocks, minimizing periodic flicker described
    in the blog.
-   **Where**: VAE decode/cache handling in `_decode_block` within
    `self_forcing.py`.

### Real-Time Prompt Updates

-   **What**: Change the prompt mid-generation; the new condition takes effect
    on the next block. Combine with a purge for faster, cleaner transitions.
-   **Where**: `SelfForcingEndlessGenerator.set_prompt(prompt)`; Socket.IO event
    `new_prompt`.

### Demo Refactor: Generator Class API

-   **What**: The previous monolithic demo flow was refactored into a reusable
    class.
-   **Where**: `self_forcing.py` exposes `SelfForcingEndlessGenerator`, used by
    `server.py`.
-   **Key methods**:
    -   `endless_generation_task(prompt, seed)`
    -   `set_prompt(prompt)`
    -   `purge_cache(purge_strength)`
    -   `stop_generating()`
-   **Streaming frames**: Frames are emitted via a queue and Socket.IO
    (`frame_ready`) for immediate playback in the UI.

For a deeper dive into motivations, trade-offs, and usage patterns (e.g., guided
transitions vs. hard prompt switches, synchronization effects, and degradation
recovery), see the blog post:
[Self-Forcing: Making AI Video Generation Endless](https://derewah.dev/projects/self-forcing-endless).

## Training

### Download text prompts and ODE initialized checkpoint

```
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
```

Note: Our training algorithm (except for the GAN version) is data-free (**no
video data is needed**). For now, we directly provide the ODE initialization
checkpoint and will add more instructions on how to perform ODE initialization
in the future (which is identical to the process described in the
[CausVid](https://github.com/tianweiy/CausVid) repo).

### Self Forcing Training with DMD

```
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_dmd \
  --disable-wandb
```

Our training run uses 600 iterations and completes in under 2 hours using 64
H100 GPUs. By implementing gradient accumulation, it should be possible to
reproduce the results in less than 16 hours using 8 H100 GPUs.

## Acknowledgements

This codebase is built on top of the open-source implementation of
[CausVid](https://github.com/tianweiy/CausVid) by
[Tianwei Yin](https://tianweiy.github.io/) and the
[Wan2.1](https://github.com/Wan-Video/Wan2.1) repo.

## Citation

If you find this codebase useful for your research, please kindly cite our
paper:

```
@article{huang2025selfforcing,
  title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal={arXiv preprint arXiv:2506.08009},
  year={2025}
}
```
