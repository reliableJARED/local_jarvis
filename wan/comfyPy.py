"""
Wan2.2 I2V Engine - Option B Implementation
============================================
Clean Python/Diffusers implementation for 16GB consumer GPUs

Features:
- Bitsandbytes NF4/INT8 quantization (fits in 16GB VRAM)
- Robust multi-format LoRA converter (ComfyUI, Kohya, LightX2V, diffusers)
- LightX2V 4-step distillation support
- Stacking multiple LoRAs (distillation + style)

Author: Built for consumer GPU video generation
"""

import torch
import gc
import re
from PIL import Image
from pathlib import Path
from typing import Optional, Union, Dict, List, Literal
from collections import OrderedDict
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers import (
    WanImageToVideoPipeline,
    WanTransformer3DModel,
    BitsAndBytesConfig,
)
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from huggingface_hub import hf_hub_download
import safetensors.torch


# =============================================================================
# LORA KEY CONVERTER - Handles multiple formats
# =============================================================================

class WanLoRAConverter:
    """
    Converts LoRA state dicts from various formats to diffusers format.
    
    Supported source formats:
    - ComfyUI/Kijai: diffusion_model.blocks.X.self_attn.q.lora_down.weight
    - Kohya/Musubi: lora_unet_blocks_X_self_attn_q.lora_down.weight  
    - LightX2V: blocks.X.self_attn.q.lora_down.weight
    - Diffusers native: transformer.blocks.X.attn1.to_q.lora_A.weight
    
    Target format (diffusers):
    - blocks.X.attn1.to_q.lora_A.weight / lora_B.weight
    - blocks.X.attn1.to_k.lora_A.weight / lora_B.weight
    - blocks.X.attn1.to_v.lora_A.weight / lora_B.weight
    - blocks.X.attn1.to_out.0.lora_A.weight / lora_B.weight
    - blocks.X.attn2.to_q.lora_A.weight / lora_B.weight (cross-attention)
    - blocks.X.ffn.0.lora_A.weight / lora_B.weight (feed-forward)
    - etc.
    """
    
    # Mapping from various source naming conventions to diffusers
    ATTENTION_MAP = {
        # Self-attention (attn1)
        'self_attn.q': 'attn1.to_q',
        'self_attn.k': 'attn1.to_k', 
        'self_attn.v': 'attn1.to_v',
        'self_attn.o': 'attn1.to_out.0',
        'self_attn.out': 'attn1.to_out.0',
        'self_attn.proj': 'attn1.to_out.0',
        # Cross-attention (attn2) 
        'cross_attn.q': 'attn2.to_q',
        'cross_attn.k': 'attn2.to_k',
        'cross_attn.v': 'attn2.to_v',
        'cross_attn.o': 'attn2.to_out.0',
        'cross_attn.out': 'attn2.to_out.0',
        'cross_attn.proj': 'attn2.to_out.0',
        # Alternative naming
        'attn.q': 'attn1.to_q',
        'attn.k': 'attn1.to_k',
        'attn.v': 'attn1.to_v',
        'attn.out': 'attn1.to_out.0',
        'attn.proj': 'attn1.to_out.0',
        # QKV combined (need special handling)
        'self_attn.qkv': None,  # Will be split
        'attn.qkv': None,
    }
    
    FFN_MAP = {
        'ffn.0': 'ffn.0',
        'ffn.2': 'ffn.2', 
        'mlp.fc1': 'ffn.0',
        'mlp.fc2': 'ffn.2',
        'ff.net.0': 'ffn.0',
        'ff.net.2': 'ffn.2',
    }
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def detect_format(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Detect the format of the LoRA state dict."""
        keys = list(state_dict.keys())
        sample_key = keys[0] if keys else ""
        
        if sample_key.startswith("diffusion_model."):
            return "comfyui"
        elif sample_key.startswith("lora_unet_"):
            return "kohya"
        elif sample_key.startswith("transformer."):
            return "diffusers"
        elif "lora_down" in sample_key or "lora_up" in sample_key:
            # Generic non-diffusers format
            if "blocks." in sample_key:
                return "lightx2v"
            return "generic"
        elif "lora_A" in sample_key or "lora_B" in sample_key:
            return "diffusers"
        else:
            return "unknown"
    
    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert LoRA state dict to diffusers format.
        
        Args:
            state_dict: Input LoRA weights
            
        Returns:
            Converted state dict in diffusers format
        """
        format_type = self.detect_format(state_dict)
        
        if self.verbose:
            print(f"  Detected LoRA format: {format_type}")
        
        if format_type == "diffusers":
            # Already in correct format, just clean up prefixes
            return self._clean_diffusers_keys(state_dict)
        elif format_type == "comfyui":
            return self._convert_comfyui(state_dict)
        elif format_type == "kohya":
            return self._convert_kohya(state_dict)
        elif format_type == "lightx2v":
            return self._convert_lightx2v(state_dict)
        elif format_type == "generic":
            return self._convert_generic(state_dict)
        else:
            if self.verbose:
                print(f"  Warning: Unknown format, attempting generic conversion")
            return self._convert_generic(state_dict)
    
    def _clean_diffusers_keys(self, state_dict: Dict) -> Dict:
        """Remove 'transformer.' prefix if present."""
        converted = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("transformer."):
                new_key = new_key[len("transformer."):]
            converted[new_key] = value
        return converted
    
    def _convert_comfyui(self, state_dict: Dict) -> Dict:
        """
        Convert ComfyUI/Kijai format to diffusers.
        
        ComfyUI format: diffusion_model.blocks.X.self_attn.q.lora_down.weight
        Diffusers format: blocks.X.attn1.to_q.lora_A.weight
        """
        converted = {}
        
        for key, value in state_dict.items():
            new_key = self._convert_single_key_comfyui(key)
            if new_key:
                converted[new_key] = value
            elif self.verbose and "alpha" not in key.lower():
                print(f"    Skipped key: {key}")
                
        return converted
    
    def _convert_single_key_comfyui(self, key: str) -> Optional[str]:
        """Convert a single ComfyUI key to diffusers format."""
        # Remove diffusion_model prefix
        if key.startswith("diffusion_model."):
            key = key[len("diffusion_model."):]
        
        # Handle lora_down -> lora_A, lora_up -> lora_B
        key = key.replace(".lora_down.", ".lora_A.")
        key = key.replace(".lora_up.", ".lora_B.")
        
        # Convert attention naming
        for src, dst in self.ATTENTION_MAP.items():
            if dst and src in key:
                key = key.replace(src, dst)
                break
        
        # Convert FFN naming
        for src, dst in self.FFN_MAP.items():
            if src in key:
                key = key.replace(src, dst)
                break
                
        return key
    
    def _convert_kohya(self, state_dict: Dict) -> Dict:
        """
        Convert Kohya/Musubi format to diffusers.
        
        Kohya format: lora_unet_blocks_X_self_attn_q.lora_down.weight
        """
        converted = {}
        
        for key, value in state_dict.items():
            new_key = self._convert_single_key_kohya(key)
            if new_key:
                converted[new_key] = value
                
        return converted
    
    def _convert_single_key_kohya(self, key: str) -> Optional[str]:
        """Convert a single Kohya key to diffusers format."""
        # Remove lora_unet_ prefix and convert underscores to dots
        if key.startswith("lora_unet_"):
            key = key[len("lora_unet_"):]
        
        # Convert lora naming
        key = key.replace(".lora_down.", ".lora_A.")
        key = key.replace(".lora_up.", ".lora_B.")
        
        # Convert underscores to dots for structure
        # But be careful with lora_A and lora_B
        parts = key.split(".")
        if len(parts) >= 2:
            # Reconstruct with proper structure
            prefix = parts[0].replace("_", ".")
            suffix = ".".join(parts[1:])
            key = f"{prefix}.{suffix}"
        
        # Apply attention and FFN mappings
        for src, dst in self.ATTENTION_MAP.items():
            if dst and src.replace(".", "_") in key or src in key:
                key = key.replace(src.replace(".", "_"), dst)
                key = key.replace(src, dst)
                break
                
        for src, dst in self.FFN_MAP.items():
            if src in key:
                key = key.replace(src, dst)
                break
                
        return key
    
    def _convert_lightx2v(self, state_dict: Dict) -> Dict:
        """
        Convert LightX2V format to diffusers.
        
        LightX2V format: blocks.X.self_attn.q.lora_down.weight
        """
        converted = {}
        
        for key, value in state_dict.items():
            new_key = key
            
            # Convert lora naming
            new_key = new_key.replace(".lora_down.", ".lora_A.")
            new_key = new_key.replace(".lora_up.", ".lora_B.")
            
            # Apply attention mappings
            for src, dst in self.ATTENTION_MAP.items():
                if dst and src in new_key:
                    new_key = new_key.replace(src, dst)
                    break
            
            # Apply FFN mappings
            for src, dst in self.FFN_MAP.items():
                if src in new_key:
                    new_key = new_key.replace(src, dst)
                    break
            
            converted[new_key] = value
            
        return converted
    
    def _convert_generic(self, state_dict: Dict) -> Dict:
        """Fallback generic conversion."""
        converted = {}
        
        for key, value in state_dict.items():
            new_key = key
            
            # Remove common prefixes
            for prefix in ["diffusion_model.", "transformer.", "model.", "lora_unet_"]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            
            # Convert lora naming
            new_key = new_key.replace(".lora_down.", ".lora_A.")
            new_key = new_key.replace(".lora_up.", ".lora_B.")
            new_key = new_key.replace("_lora_down_", ".lora_A.")
            new_key = new_key.replace("_lora_up_", ".lora_B.")
            
            # Try attention mappings
            for src, dst in self.ATTENTION_MAP.items():
                if dst and src in new_key:
                    new_key = new_key.replace(src, dst)
                    break
            
            converted[new_key] = value
            
        return converted


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================

class Wan22I2VEngine:
    """
    Wan2.2 I2V Engine with bitsandbytes quantization and multi-format LoRA support.
    
    Designed for 16GB consumer GPUs with features:
    - NF4/INT8 quantization via bitsandbytes
    - LightX2V 4-step distillation support
    - Multiple LoRA stacking (distillation + style)
    - CPU offloading for memory efficiency
    """
    
    def __init__(
        self,
        quantization: Literal["nf4", "int8", "none"] = "nf4",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the Wan2.2 I2V engine.
        
        Args:
            quantization: Quantization method - "nf4" (4-bit), "int8" (8-bit), or "none"
            device: Device to use ("cuda" or "cpu")
            dtype: Data type for non-quantized layers (bfloat16 recommended)
        """
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        self.loaded_loras: List[str] = []
        self.lora_converter = WanLoRAConverter(verbose=True)
        
        print("=" * 70)
        print("Wan2.2 I2V Engine - Option B (Quantized + LoRA Converter)")
        print("=" * 70)
        print(f"  Quantization: {quantization}")
        print(f"  Device: {device}")
        print(f"  Dtype: {dtype}")
        print()
        
        self._load_pipeline()
        
    def _load_pipeline(self):
        """Load the pipeline with quantization."""
        
        # Setup quantization config
        quant_config = None
        if self.quantization == "nf4":
            print("1. Setting up NF4 quantization (4-bit)...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,  # Nested quantization for extra savings
            )
        elif self.quantization == "int8":
            print("1. Setting up INT8 quantization (8-bit)...")
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:
            print("1. No quantization (full precision)...")
        
        # Load transformers with quantization
        print("\n2. Loading High-Noise Transformer...")
        if quant_config:
            transformer_high = WanTransformer3DModel.from_pretrained(
                "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=self.dtype,
            )
        else:
            transformer_high = WanTransformer3DModel.from_pretrained(
                "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                subfolder="transformer",
                torch_dtype=self.dtype,
            )
        
        print("3. Loading Low-Noise Transformer...")
        if quant_config:
            transformer_low = WanTransformer3DModel.from_pretrained(
                "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                subfolder="transformer_2",
                quantization_config=quant_config,
                torch_dtype=self.dtype,
            )
        else:
            transformer_low = WanTransformer3DModel.from_pretrained(
                "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                subfolder="transformer_2",
                torch_dtype=self.dtype,
            )
        
        print("\n4. Building Pipeline...")
        # Load pipeline WITHOUT transformers (pass None to skip loading them)
        # This loads image_encoder, image_processor, vae, text_encoder, tokenizer, scheduler
        # Load Image Encoder and Processor explicitly
        print(">> Loading Image Encoder (CLIP-ViT-H-14)...")
        # Loads image_encoder and image_processor from Wan2.1
        image_encoder = CLIPVisionModel.from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            subfolder="image_encoder",
            torch_dtype=torch.float32
        )

        image_processor = CLIPImageProcessor.from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            subfolder="image_processor"
        )

        self.pipe = WanImageToVideoPipeline.from_pretrained(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            transformer=None,  # Skip loading - we'll add our quantized one
            torch_dtype=self.dtype,
            image_encoder=image_encoder,
            image_processor=image_processor,
            
        )
        
        # Now set our quantized transformers
        print("   Setting quantized transformers...")
        self.pipe.transformer = transformer_high
        self.pipe.transformer_2 = transformer_low
        
        # Register transformer_2 for CPU offloading to work properly
        self.pipe.register_modules(transformer_2=transformer_low)
        
        print("\n5. Enabling memory optimizations...")
        self.pipe.enable_model_cpu_offload()
        #self.pipe.vae.enable_tiling()
        
        # Store references
        self.transformer_high = self.pipe.transformer
        self.transformer_low = self.pipe.transformer_2
        
        print("\n" + "=" * 70)
        print("Engine Ready!")
        print("=" * 70 + "\n")
    
    def load_lightx2v_distillation(
        self,
        strength_high: float = 1.0,
        strength_low: float = 1.0,
        num_steps: int = 4,
    ):
        """
        Load LightX2V 4-step distillation LoRAs for fast inference.
        
        Args:
            strength_high: LoRA strength for high-noise transformer
            strength_low: LoRA strength for low-noise transformer
            num_steps: Number of inference steps (4 recommended for distillation)
        """
        print("Loading LightX2V 4-step distillation LoRAs...")
        
        # Download the distillation LoRAs
        high_lora_path = hf_hub_download(
            repo_id="lightx2v/Wan2.2-Distill-Loras",
            filename="wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
        )
        low_lora_path = hf_hub_download(
            repo_id="lightx2v/Wan2.2-Distill-Loras",
            filename="wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
        )
        
        # Load and apply
        self._load_lora_to_transformer(
            self.transformer_high, 
            high_lora_path, 
            strength_high, 
            "lightx2v_high"
        )
        self._load_lora_to_transformer(
            self.transformer_low,
            low_lora_path,
            strength_low,
            "lightx2v_low"
        )
        
        # Configure scheduler for 4-step inference
        print(f"  Configuring scheduler for {num_steps}-step inference...")
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            flow_shift=8.0,  # Recommended for LightX2V
        )
        
        self.loaded_loras.append("lightx2v_distillation")
        self.default_steps = num_steps
        print(f"  ✓ LightX2V distillation loaded (recommended steps: {num_steps})")
    
    def load_lora(
        self,
        lora_path_or_repo: str,
        filename: Optional[str] = None,
        strength_high: float = 1.0,
        strength_low: float = 0.8,
        adapter_name: Optional[str] = None,
        apply_to: Literal["both", "high", "low"] = "both",
    ):
        """
        Load a style/content LoRA from local path or HuggingFace Hub.
        
        Args:
            lora_path_or_repo: Local file path OR HuggingFace repo ID
            filename: Filename if loading from HuggingFace repo
            strength_high: LoRA strength for high-noise transformer
            strength_low: LoRA strength for low-noise transformer
            adapter_name: Name for this adapter (auto-generated if None)
            apply_to: Which transformer(s) to apply to ("both", "high", "low")
        """
        # Determine if local path or hub
        if filename is not None:
            # Hub download
            print(f"Downloading LoRA from Hub: {lora_path_or_repo}/{filename}")
            lora_path = hf_hub_download(
                repo_id=lora_path_or_repo,
                filename=filename,
            )
        else:
            # Local path
            lora_path = lora_path_or_repo
            print(f"Loading LoRA from: {lora_path}")
        
        if adapter_name is None:
            adapter_name = Path(lora_path).stem
        
        # Apply to specified transformers
        if apply_to in ["both", "high"]:
            self._load_lora_to_transformer(
                self.transformer_high,
                lora_path,
                strength_high,
                f"{adapter_name}_high"
            )
        
        if apply_to in ["both", "low"]:
            self._load_lora_to_transformer(
                self.transformer_low,
                lora_path,
                strength_low,
                f"{adapter_name}_low"
            )
        
        self.loaded_loras.append(adapter_name)
        print(f"  ✓ LoRA '{adapter_name}' loaded")
    
    def load_lora_pair(
        self,
        repo_id: str,
        high_filename: str,
        low_filename: str,
        strength_high: float = 1.0,
        strength_low: float = 1.0,
        adapter_name: Optional[str] = None,
    ):
        """
        Load separate high/low noise LoRAs (for paired LoRA releases).
        
        Args:
            repo_id: HuggingFace repo ID
            high_filename: Filename for high-noise LoRA
            low_filename: Filename for low-noise LoRA
            strength_high: Strength for high-noise transformer
            strength_low: Strength for low-noise transformer
            adapter_name: Name for this adapter pair
        """
        print(f"Loading LoRA pair from: {repo_id}")
        
        high_path = hf_hub_download(repo_id=repo_id, filename=high_filename)
        low_path = hf_hub_download(repo_id=repo_id, filename=low_filename)
        
        if adapter_name is None:
            adapter_name = Path(high_filename).stem.replace("-high", "").replace("_high", "")
        
        self._load_lora_to_transformer(
            self.transformer_high,
            high_path,
            strength_high,
            f"{adapter_name}_high"
        )
        self._load_lora_to_transformer(
            self.transformer_low,
            low_path,
            strength_low,
            f"{adapter_name}_low"
        )
        
        self.loaded_loras.append(adapter_name)
        print(f"  ✓ LoRA pair '{adapter_name}' loaded")
    
    def _load_lora_to_transformer(
        self,
        transformer: WanTransformer3DModel,
        lora_path: str,
        strength: float,
        adapter_name: str,
    ):
        """Internal: Load and apply LoRA to a single transformer."""
        
        # Load state dict
        state_dict = safetensors.torch.load_file(lora_path)
        
        # Convert to diffusers format
        converted_state_dict = self.lora_converter.convert(state_dict)
        
        if not converted_state_dict:
            print(f"    Warning: No keys converted for {adapter_name}")
            return
        
        # Try to load via PEFT adapter
        try:
            transformer.load_lora_adapter(
                converted_state_dict,
                adapter_name=adapter_name,
            )
            transformer.set_adapters([adapter_name], weights=[strength])
            print(f"    ✓ Applied {adapter_name} (strength={strength})")
        except Exception as e:
            print(f"    Warning: PEFT adapter failed: {e}")
            print(f"    Attempting direct weight merge...")
            self._merge_lora_weights(transformer, converted_state_dict, strength)
    
    def _merge_lora_weights(
        self,
        transformer: WanTransformer3DModel,
        lora_state_dict: Dict[str, torch.Tensor],
        strength: float,
    ):
        """
        Fallback: Directly merge LoRA weights into model.
        
        This is a fallback when PEFT adapter loading fails.
        LoRA formula: W_new = W_original + strength * (lora_B @ lora_A)
        """
        merged_count = 0
        
        # Group lora_A and lora_B pairs
        lora_pairs = {}
        for key in lora_state_dict.keys():
            if ".lora_A." in key:
                base_key = key.replace(".lora_A.", ".")
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["A"] = key
            elif ".lora_B." in key:
                base_key = key.replace(".lora_B.", ".")
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["B"] = key
        
        # Apply each pair
        for base_key, pair in lora_pairs.items():
            if "A" not in pair or "B" not in pair:
                continue
            
            lora_A = lora_state_dict[pair["A"]]
            lora_B = lora_state_dict[pair["B"]]
            
            # Find corresponding weight in model
            # Convert base_key to model parameter path
            param_key = base_key.rstrip(".") + ".weight"
            
            # Try to find the parameter
            try:
                # Navigate to the parameter
                parts = param_key.split(".")
                module = transformer
                for part in parts[:-1]:
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)
                
                param_name = parts[-1]
                if hasattr(module, param_name):
                    original_weight = getattr(module, param_name)
                    
                    # Compute LoRA delta: lora_B @ lora_A
                    delta = (lora_B @ lora_A).to(original_weight.dtype).to(original_weight.device)
                    
                    # Apply with strength
                    with torch.no_grad():
                        original_weight.add_(strength * delta)
                    
                    merged_count += 1
            except Exception:
                continue
        
        print(f"    Merged {merged_count} LoRA weight pairs directly")
    
    def generate(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        negative_prompt: str = "low quality, blurry, distortion, watermark, static, worst quality",
        num_frames: int = 81,
        width: int = 832,
        height: int = 480,
        num_steps: Optional[int] = None,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> List[Image.Image]:
        """
        Generate video from image + prompt.
        
        Args:
            image: Input image (path or PIL Image)
            prompt: Text prompt describing desired motion/content
            negative_prompt: Negative prompt
            num_frames: Number of frames to generate (default 81 = ~5 seconds @ 16fps)
            width: Output width (832 for 480p, 1280 for 720p)
            height: Output height (480 for 480p, 720 for 720p)
            num_steps: Inference steps (4 with LightX2V, 20+ without)
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            output_path: If provided, save video to this path
            
        Returns:
            List of PIL Images (video frames)
        """
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        
        # Use default steps if distillation loaded
        if num_steps is None:
            num_steps = getattr(self, 'default_steps', 20)
        
        # Load and preprocess image
        if isinstance(image, str):
            image = load_image(image).convert("RGB")
        
        image = self._smart_resize(image, width, height)
        
        # Setup generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        
        # Log generation params
        print(f"\n{'='*60}")
        print("Generating Video")
        print(f"{'='*60}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Frames: {num_frames}")
        print(f"  Steps: {num_steps}")
        print(f"  Guidance: {guidance_scale}")
        print(f"  Seed: {seed}")
        print(f"  LoRAs: {self.loaded_loras if self.loaded_loras else 'None'}")
        print(f"  Prompt: {prompt[:80]}...")
        print()
        
        # Generate
        output = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        frames = output.frames[0]
        
        # Save if requested
        if output_path:
            export_to_video(frames, output_path, fps=16)
            print(f"✓ Saved to: {output_path}")
        
        print(f"✓ Generation complete!")
        return frames
    
    def _smart_resize(self, image: Image.Image, w: int, h: int) -> Image.Image:
        """Center-crop resize to target dimensions."""
        img_ratio = image.width / image.height
        target_ratio = w / h
        
        if img_ratio > target_ratio:
            # Image is wider - fit height, crop width
            new_h = h
            new_w = int(h * img_ratio)
        else:
            # Image is taller - fit width, crop height
            new_w = w
            new_h = int(w / img_ratio)
        
        # Resize
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center crop
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        return image.crop((left, top, left + w, top + h))


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    # Example 1: Fast generation with LightX2V distillation (4 steps)
    print("\n" + "="*70)
    print("EXAMPLE: Wan2.2 I2V with LightX2V 4-step + Custom Style LoRA")
    print("="*70 + "\n")
    
    # Initialize with NF4 quantization (fits in 16GB VRAM)
    engine = Wan22I2VEngine(quantization="nf4")
    
    # Load LightX2V distillation for 4-step inference
    engine.load_lightx2v_distillation(
        strength_high=1.0,
        strength_low=1.0,
        num_steps=4,
    )
    
    # Stack a custom style LoRA on top (example - uncomment to use)
    # engine.load_lora(
    #     lora_path_or_repo="your-username/your-lora-repo",
    #     filename="your_style_lora.safetensors",
    #     strength_high=0.8,
    #     strength_low=0.6,
    # )
    
    # Or load from local path:
    # engine.load_lora(
    #     lora_path_or_repo="/path/to/your/style_lora.safetensors",
    #     strength_high=0.8,
    #     strength_low=0.6,
    # )
    
    # Generate video
    frames = engine.generate(
        image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
        prompt="The astronaut walks forward confidently, camera follows smoothly, cinematic motion",
        num_frames=81,
        width=832,
        height=480,
        num_steps=4,  # 4 steps with LightX2V!
        guidance_scale=5.0,
        seed=42,
        output_path="output_lightx2v.mp4",
    )