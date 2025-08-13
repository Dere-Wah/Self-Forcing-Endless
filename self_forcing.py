import base64
import os
import queue
import urllib.request
import uuid
from io import BytesIO
from threading import Event, Thread
from typing import Optional, Callable, Any
from demo_utils.vae_block3 import VAEDecoderWrapper
import numpy as np
import torch
from PIL import Image
from demo_utils.constant import ZERO_VAE_CACHE
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, \
    move_model_to_device_with_memory_preservation

from pipeline import CausalInferencePipeline

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder

def initialize_vae_decoder(use_taehv=False, use_trt=False):
    """Initialize VAE decoder based on the selected option"""

    if use_trt:
        from demo_utils.vae import VAETRTWrapper
        current_vae_decoder = VAETRTWrapper()
        return current_vae_decoder

    if use_taehv:
        from demo_utils.taehv import TAEHV
        # Check if taew2_1.pth exists in checkpoints folder, download if missing
        taehv_checkpoint_path = "checkpoints/taew2_1.pth"
        if not os.path.exists(taehv_checkpoint_path):
            print(f"taew2_1.pth not found in checkpoints folder {taehv_checkpoint_path}. Downloading...")
            os.makedirs("checkpoints", exist_ok=True)
            download_url = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"
            try:
                urllib.request.urlretrieve(download_url, taehv_checkpoint_path)
                print(f"Successfully downloaded taew2_1.pth to {taehv_checkpoint_path}")
            except Exception as e:
                print(f"Failed to download taew2_1.pth: {e}")
                raise

        class DotDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        class TAEHVDiffusersWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dtype = torch.float16
                self.taehv = TAEHV(checkpoint_path=taehv_checkpoint_path).to(self.dtype)
                self.config = DotDict(scaling_factor=1.0)

            def decode(self, latents, return_dict=None):
                # n, c, t, h, w = latents.shape
                # low-memory, set parallel=True for faster + higher memory
                return self.taehv.decode_video(latents, parallel=False).mul_(2).sub_(1)

        current_vae_decoder = TAEHVDiffusersWrapper()
    else:
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load('wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth', map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)

    current_vae_decoder.eval()
    current_vae_decoder.to(dtype=torch.float16)
    current_vae_decoder.requires_grad_(False)
    current_vae_decoder.to(gpu)
    current_use_taehv = use_taehv

    print(f"✅ VAE decoder initialized with {'TAEHV' if use_taehv else 'default VAE'}")
    return current_vae_decoder


def tensor_to_base_64_frame(frame_tensor: torch.Tensor):
    """Convert a single frame tensor to base64 image string."""
    # Clamp and normalize to 0-255
    frame = torch.clamp(frame_tensor.float(), -1., 1.) * 127.5 + 127.5
    frame = frame.to(torch.uint8).cpu().numpy()

    # CHW -> HWC
    if len(frame.shape) == 3:
        frame = np.transpose(frame, (1, 2, 0))

    # Convert to PIL Image
    if frame.shape[2] == 3:  # RGB
        image = Image.fromarray(frame, 'RGB')
    else:  # Handle other formats
        image = Image.fromarray(frame)

    # Convert to base64
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=100)
    img_uuid = uuid.uuid4()
    if not os.path.exists("./images/"):
        os.makedirs("./images/")
    image.save("./images/%s.jpg" % img_uuid)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


class SelfForcingEndlessGenerator:
    def __init__(self, checkpoint_path: str, config, trt=False):
        self.do_purge_cache = False
        self.purge_strength = 3  # Default to last index (original behavior)
        self.cleaning_up = False
        self.low_memory = get_cuda_free_memory_gb(gpu) < 40
        self.text_encoder = WanTextEncoder()
        self.transformer = WanDiffusionWrapper(is_causal=True)
        self.trt = trt
        self.current_use_taehv = False
        # Initialize with default VAE
        self.vae_decoder = initialize_vae_decoder(use_taehv=self.current_use_taehv, use_trt=trt)

        # Set a reasonable local attention window size (e.g. 20 blocks)
        self.transformer = WanDiffusionWrapper(is_causal=True, local_attn_size=20)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.transformer.load_state_dict(state_dict['generator_ema'])

        self.text_encoder.eval()
        self.transformer.eval()

        self.transformer.to(dtype=torch.float16)
        self.text_encoder.to(dtype=torch.bfloat16)

        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

        self.pipeline = CausalInferencePipeline(
            config,
            device=gpu,
            generator=self.transformer,
            text_encoder=self.text_encoder,
            vae=self.vae_decoder
        )

        if self.low_memory:
            DynamicSwapInstaller.install_model(self.text_encoder, device=gpu)
        else:
            self.text_encoder.to(gpu)
        self.transformer.to(gpu)

        # Initialize generator state
        self.generation_active = False
        self.image_queue = queue.Queue()
        self.seed = -1
        self.block_idx = 0
        self.conditional_dict = None
        
        # Rolling generation parameters
        self.current_start_frame = 0
        self.num_input_frames = 0
        self.vae_cache = ZERO_VAE_CACHE
        for i in range(len(self.vae_cache)):
            self.vae_cache[i] = self.vae_cache[i].to(device=gpu, dtype=torch.float16)

        self.rnd = None
        self.noise = None
        
        # Initialize KV cache once at the start
        self._initialize_caches()

    def _initialize_caches(self):
        """Initialize KV cache and cross-attention cache once."""
        self.pipeline._initialize_kv_cache(batch_size=1, dtype=torch.float16, device=gpu)
        self.pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.float16, device=gpu)
        print(f"✅ Initialized KV cache with size: {self.pipeline.kv_cache1[0]['k'].shape[1]} tokens")
        print(f"✅ Local attention size: {self.pipeline.local_attn_size} frames")

    def _decode_block(self, idx, denoised_pred):
        """
        Decodes a latent block into frames, using the VAE decoder.
        This also updates the VAE cache with the current block that is being decoded for the
        decoding of the next temporal block.
        """
        if self.trt:
            all_current_pixels = []
            for i in range(denoised_pred.shape[1]):
                is_first_frame = torch.tensor(1.0).cuda().half() if idx == 0 and i == 0 else \
                    torch.tensor(0.0).cuda().half()
                outputs = self.vae_decoder.forward(denoised_pred[:, i:i + 1, :, :, :].half(), is_first_frame,
                                                   *self.vae_cache)
                # outputs = vae_decoder.forward(denoised_pred.float(), *vae_cache)
                current_pixels, vae_cache = outputs[0], outputs[1:]
                print(current_pixels.max(), current_pixels.min())
                all_current_pixels.append(current_pixels.clone())
            pixels = torch.cat(all_current_pixels, dim=1)
            if idx == 0:
                pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block
        else:
            if self.current_use_taehv:
                if self.vae_cache is None:
                    self.vae_cache = denoised_pred
                else:
                    denoised_pred = torch.cat([self.vae_cache, denoised_pred], dim=1)
                    self.vae_cache = denoised_pred[:, -3:, :, :, :]
                pixels = self.vae_decoder.decode(denoised_pred)
                print(f"denoised_pred shape: {denoised_pred.shape}")
                print(f"pixels shape: {pixels.shape}")
                if idx == 0:
                    pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block
                else:
                    pixels = pixels[:, 12:, :, :, :]

            else:
                pixels, self.vae_cache = self.vae_decoder(denoised_pred.half(), *self.vae_cache)
                if idx == 0:
                    pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block
        return pixels

    def _denoise_frame(self,
                       current_timestep: int,
                       index: int,
                       noisy_input,
                       current_conditional_dict,
                       current_num_frames: int):
        timestep = torch.ones([1, current_num_frames], device=gpu,
                              dtype=torch.int64) * current_timestep
        """
        Denoise frames inside of a block. Each call to this method is a denoising step.
        The function returns the denoised block and the noisy input for the following denoising step.
        """
        print(f"denoising step: {current_timestep}")
        if self.do_purge_cache and current_timestep == self.pipeline.denoising_step_list[-(self.purge_strength + 1)]:
            self._perform_purge_cache()

        _, denoised_pred = self.transformer(
            noisy_image_or_video=noisy_input,
            conditional_dict=current_conditional_dict,
            timestep=timestep,
            kv_cache=self.pipeline.kv_cache1,
            crossattn_cache=self.pipeline.crossattn_cache,
            current_start=self.current_start_frame * self.pipeline.frame_seq_length
        )

        if index < len(self.pipeline.denoising_step_list) - 1:
            # If this is not the last denoising timestep, update noise
            next_timestep = self.pipeline.denoising_step_list[index + 1]
            noisy_input = self.pipeline.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                next_timestep * torch.ones([1 * current_num_frames], device=gpu, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
        return denoised_pred, noisy_input, timestep

    def _generate_block(self, idx: int) -> torch.Tensor:
        """
        Generate a block. Each block is a latent which contains 3 frames.
        The KV cache automatically handles the rolling window mechanism.
        """
        print(f"[Block {self.block_idx}] Generating at frame position {self.current_start_frame}")
        
        # Check KV cache state
        if self.block_idx > 0 and self.block_idx % 10 == 0:
            global_end = self.pipeline.kv_cache1[0]['global_end_index'].item()
            local_end = self.pipeline.kv_cache1[0]['local_end_index'].item()
            print(f"  KV Cache: global_end={global_end}, local_end={local_end}")
        
        current_conditional_dict = self.conditional_dict
        current_num_frames = self.pipeline.num_frame_per_block

        # Generate noise for the current block
        noise_idx = self.current_start_frame - self.num_input_frames
        noisy_input = self.noise[:, noise_idx:noise_idx + current_num_frames]

        for index, current_timestep in enumerate(self.pipeline.denoising_step_list):
            denoised_pred, new_noisy_input, timestep = self._denoise_frame(current_timestep,
                                                                           index,
                                                                           noisy_input,
                                                                           current_conditional_dict,
                                                                           current_num_frames)
            noisy_input = new_noisy_input

        # Update KV cache with the clean denoised result
        self.transformer(
            noisy_image_or_video=denoised_pred,
            conditional_dict=current_conditional_dict,
            timestep=torch.zeros_like(timestep),
            kv_cache=self.pipeline.kv_cache1,
            crossattn_cache=self.pipeline.crossattn_cache,
            current_start=self.current_start_frame * self.pipeline.frame_seq_length,
        )
        
        # Decode latent into pixel tensors
        pixels = self._decode_block(idx, denoised_pred)
        block_frames = pixels.shape[1]
        # Send frames in the form of tensors to the queue
        self._enqueue_tensors(block_frames, pixels)
        
        # Update position for next block
        self.current_start_frame += self.pipeline.num_frame_per_block
        self.block_idx += 1
        
        return denoised_pred

    def _enqueue_tensors(self, block_frames, pixels):
        """
        Converts a tensor into a base64 representation frame and enqueues it.
        """
        for frame_idx in range(block_frames):
            frame_tensor = pixels[0, frame_idx].cpu()
            base64_frame = tensor_to_base_64_frame(frame_tensor)
            self.image_queue.put(base64_frame)

    def _encode_prompt(self, prompt: str):
        """
        Encodes a textual prompt into a latent.
        """
        # Text encoding
        print(f"Encoding prompt: {prompt}")
        conditional_dict = self.text_encoder(text_prompts=[prompt])
        for key, value in conditional_dict.items():
            conditional_dict[key] = value.to(dtype=torch.float16)
        if self.low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
        return conditional_dict

    def _reset_generator(self):
        """
        Resets the generator after an endless generation is stopped.
        """
        self.image_queue.empty()
        self.vae_cache = ZERO_VAE_CACHE
        for i in range(len(self.vae_cache)):
            self.vae_cache[i] = self.vae_cache[i].to(device=gpu, dtype=torch.float16)
        self.block_idx = 0
        self.current_start_frame = 0
        self.num_input_frames = 0
        
        # Reset KV cache indices but keep the allocated memory
        for block_index in range(len(self.pipeline.kv_cache1)):
            self.pipeline.kv_cache1[block_index]["global_end_index"].fill_(0)
            self.pipeline.kv_cache1[block_index]["local_end_index"].fill_(0)
        
        # Reset cross attention cache
        for block_index in range(self.pipeline.num_transformer_blocks):
            self.pipeline.crossattn_cache[block_index]["is_init"] = False

    def stop_generating(self):
        """
        Stops a generation, if active.
        """
        if self.generation_active:
            self.generation_active = False
            self.cleaning_up = True

    def set_prompt(self, prompt: str):
        """
        Updates the current prompt used for conditioning the generation.
        The prompt can be input at any moment, but will only start to affect generation
        on the next generated block.
        """
        print(f"Updating prompt: {prompt}")
        self.conditional_dict = self._encode_prompt(prompt)
        
        # Reset cross attention cache to ensure the new prompt takes effect
        for block_index in range(self.pipeline.num_transformer_blocks):
            self.pipeline.crossattn_cache[block_index]["is_init"] = False

    def endless_generation_task(self, prompt, seed):
        """
        Thread-blocking task for endless generation.
        The rolling KV cache mechanism handles infinite length automatically.
        """
        if self.cleaning_up or self.generation_active:
            print("Could not start a new generation while another one is not cleared.")
            return

        self.generation_active = True
        self._reset_generator()

        try:
            print("Starting endless generation...")
            self.conditional_dict = self._encode_prompt(prompt)
            self.seed = seed

            # Generate initial noise for a very long sequence
            # The model will use this noise buffer as it generates
            max_frames = 10000  # Large buffer for "endless" generation
            self.rnd = torch.Generator(gpu).manual_seed(self.seed)
            self.noise = torch.randn([1, max_frames, 16, 60, 104], device=gpu, dtype=torch.float16, generator=self.rnd)

            while self.generation_active:
                # Check if we're approaching the end of our noise buffer
                if self.current_start_frame >= max_frames - 100:
                    print("⚠️ Approaching noise buffer limit, stopping generation")
                    break
                
                # Generate next block using the rolling KV cache
                self._generate_block(self.block_idx)
                
        except Exception as e:
            import traceback
            print(f"❌ Error in generation thread: {e}")
            traceback.print_exc()
        finally:
            print("Generation complete.")
            self._reset_generator()
            self.cleaning_up = False
            self.generation_active = False

    def purge_cache(self, purge_strength: int = 3):
        """
        Set up cache purging with specified strength.
        Args:
            purge_strength: Integer between 0 and 3 controlling when to purge during denoising steps.
                          0 = earliest possible step, 3 = latest step (default)
        """
        if not 0 <= purge_strength <= 3:
            raise ValueError("purge_strength must be between 0 and 3")
        self.purge_strength = purge_strength
        self.do_purge_cache = True

    def _perform_purge_cache(self):
        """
        Clears the KV and cross-attention caches, and uses the latest generated latent as the new context.
        This allows for a "fresh start" in generation while maintaining continuity with the last frame.
        """
        if not self.generation_active:
            print("Cannot purge cache when generation is not active")
            return
            
        print("Purging caches and using latest latent as new context...")
        
        # Reset KV cache indices but keep the allocated memory
        for block_index in range(len(self.pipeline.kv_cache1)):
            self.pipeline.kv_cache1[block_index]["global_end_index"].fill_(0)
            self.pipeline.kv_cache1[block_index]["local_end_index"].fill_(0)
            
        # Re-initialize KV cache and cross-attention cache
        self.pipeline._initialize_kv_cache(batch_size=1, dtype=torch.float16, device=gpu)
        self.pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.float16, device=gpu)
        
        # Reset VAE cache to zeros except for the latest block
        new_vae_cache = ZERO_VAE_CACHE
        for i in range(len(new_vae_cache)):
            new_vae_cache[i] = new_vae_cache[i].to(device=gpu, dtype=torch.float16)
        self.vae_cache = new_vae_cache
        
        # Reset counters but keep the current prompt
        self.current_start_frame = 0
        self.block_idx = 0
        self.num_input_frames = 0
        
        # Generate initial noise for the first block to ensure proper initialization
        noise_idx = self.current_start_frame - self.num_input_frames
        noisy_input = self.noise[:, noise_idx:noise_idx + self.pipeline.num_frame_per_block]
        
        # Initialize the first block with the current prompt
        timestep = torch.zeros([1, self.pipeline.num_frame_per_block], device=gpu, dtype=torch.int64)
        self.transformer(
            noisy_image_or_video=noisy_input,
            conditional_dict=self.conditional_dict,
            timestep=timestep,
            kv_cache=self.pipeline.kv_cache1,
            crossattn_cache=self.pipeline.crossattn_cache,
            current_start=self.current_start_frame * self.pipeline.frame_seq_length,
        )
        self.do_purge_cache = False

