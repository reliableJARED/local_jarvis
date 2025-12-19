"""
ComfyUI to Diffusers LoRA Converter for Wan Models

This script converts ComfyUI-format Wan LoRAs to diffusers-compatible format.
Based on ComfyUI's lora.py conversion logic.

Usage:
    python convert_comfy_lora_to_diffusers.py --input comfy_lora.safetensors --output diffusers_lora.safetensors
"""

import torch
from safetensors.torch import load_file, save_file
import argparse
from pathlib import Path
import re

def convert_comfy_wan_lora_to_diffusers(input_path, output_path, verbose=False):
    """
    Convert ComfyUI Wan LoRA to diffusers format
    
    Args:
        input_path: Path to ComfyUI LoRA file
        output_path: Path to save diffusers LoRA file
        verbose: Print conversion details
    """
    
    print(f"\n=== Converting Wan LoRA ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    # Load the ComfyUI LoRA
    try:
        lora_sd = load_file(input_path)
        print(f"✓ Loaded {len(lora_sd)} keys from ComfyUI LoRA")
    except Exception as e:
        print(f"✗ Failed to load input file: {e}")
        return False
    
    # Analyze the key structure
    sample_keys = list(lora_sd.keys())[:5]
    print(f"\nSample keys from input:")
    for key in sample_keys:
        print(f"  {key}")
    
    diffusers_sd = {}
    conversion_log = []
    skipped_keys = []
    
    for key, value in lora_sd.items():
        original_key = key
        
        # Remove common prefixes
        if key.startswith("diffusion_model."):
            key = key[len("diffusion_model."):]
        elif key.startswith("model.diffusion_model."):
            key = key[len("model.diffusion_model."):]
        
        # Wan-specific conversions based on ComfyUI's lora.py
        # Pattern: blocks.X.module.submodule -> transformer.blocks.X.module.submodule
        
        converted = False
        new_key = None
        
        # Strategy 1: Standard ComfyUI format
        # diffusion_model.blocks.0.attn.to_q.lora_down.weight -> transformer.blocks.0.attn.to_q.lora_down.weight
        if "blocks." in key:
            if ".lora_up.weight" in key or ".lora_down.weight" in key or ".alpha" in key:
                new_key = f"transformer.{key}"
                converted = True
        
        # Strategy 2: Underscore format (some trainers)
        # lora_unet_blocks_0_attn_to_q -> transformer.blocks.0.attn.to_q
        elif key.startswith("lora_unet_") or key.startswith("lora_transformer_"):
            # Remove lora_unet_ or lora_transformer_ prefix
            if key.startswith("lora_unet_"):
                key = key[len("lora_unet_"):]
            elif key.startswith("lora_transformer_"):
                key = key[len("lora_transformer_"):]
            
            # Replace underscores with dots, but preserve lora_up/lora_down
            # blocks_0_attn_to_q_lora_down_weight -> blocks.0.attn.to_q.lora_down.weight
            if "_lora_up_weight" in key:
                key = key.replace("_lora_up_weight", ".lora_up.weight")
            if "_lora_down_weight" in key:
                key = key.replace("_lora_down_weight", ".lora_down.weight")
            if "_alpha" in key:
                key = key.replace("_alpha", ".alpha")
            
            # Convert remaining underscores to dots
            key = key.replace("_", ".")
            new_key = f"transformer.{key}"
            converted = True
        
        # Strategy 3: head.head pattern (your specific error)
        elif "head.head" in key:
            # head.head.lora_down.weight -> proj_out.lora_down.weight
            key = key.replace("head.head", "proj_out")
            new_key = f"transformer.{key}"
            converted = True
        
        # Strategy 4: Direct transformer keys
        elif key.startswith("transformer."):
            new_key = key
            converted = True
        
        # Strategy 5: Generic lora keys without prefix
        elif ".lora_up.weight" in key or ".lora_down.weight" in key or ".alpha" in key:
            new_key = f"transformer.{key}"
            converted = True
        
        if converted and new_key:
            diffusers_sd[new_key] = value
            conversion_log.append((original_key, new_key))
            if verbose:
                print(f"  ✓ {original_key} -> {new_key}")
        else:
            # Keep as-is if no conversion pattern matched
            diffusers_sd[original_key] = value
            skipped_keys.append(original_key)
            if verbose:
                print(f"  ~ Kept as-is: {original_key}")
    
    # Print summary
    print(f"\n=== Conversion Summary ===")
    print(f"Total keys: {len(lora_sd)}")
    print(f"Converted: {len(conversion_log)}")
    print(f"Kept as-is: {len(skipped_keys)}")
    
    if skipped_keys and not verbose:
        print(f"\nKeys kept as-is (use --verbose for full list):")
        for key in skipped_keys[:5]:
            print(f"  {key}")
        if len(skipped_keys) > 5:
            print(f"  ... and {len(skipped_keys) - 5} more")
    
    # Sample output keys
    print(f"\nSample output keys:")
    for key in list(diffusers_sd.keys())[:5]:
        print(f"  {key}")
    
    # Save the converted LoRA
    try:
        save_file(diffusers_sd, output_path)
        print(f"\n✓ Saved to: {output_path}")
        
        # Print file sizes
        input_size = Path(input_path).stat().st_size / (1024 * 1024)
        output_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  Input size:  {input_size:.2f} MB")
        print(f"  Output size: {output_size:.2f} MB")
        
        return True
    except Exception as e:
        print(f"✗ Failed to save output file: {e}")
        return False


def inspect_lora(lora_path):
    """Inspect a LoRA file to understand its structure"""
    
    print(f"\n=== Inspecting LoRA ===")
    print(f"File: {lora_path}")
    
    try:
        lora_sd = load_file(lora_path)
        print(f"✓ Loaded {len(lora_sd)} keys")
        
        # Analyze key patterns
        patterns = {}
        for key in lora_sd.keys():
            # Extract pattern (first few segments)
            parts = key.split('.')
            pattern = '.'.join(parts[:3]) if len(parts) >= 3 else key
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        print(f"\nKey patterns (top 10):")
        for pattern, count in sorted(patterns.items(), key=lambda x: -x[1])[:10]:
            print(f"  {pattern}... : {count} keys")
        
        # Check for specific markers
        print(f"\nFormat indicators:")
        print(f"  Has 'diffusion_model': {any('diffusion_model' in k for k in lora_sd.keys())}")
        print(f"  Has 'transformer': {any('transformer' in k for k in lora_sd.keys())}")
        print(f"  Has 'lora_unet': {any('lora_unet' in k for k in lora_sd.keys())}")
        print(f"  Has 'head.head': {any('head.head' in k for k in lora_sd.keys())}")
        print(f"  Has '.lora_up.weight': {any('.lora_up.weight' in k for k in lora_sd.keys())}")
        print(f"  Has '.lora_down.weight': {any('.lora_down.weight' in k for k in lora_sd.keys())}")
        
        # Sample keys
        print(f"\nFirst 10 keys:")
        for i, key in enumerate(list(lora_sd.keys())[:10]):
            shape = lora_sd[key].shape if hasattr(lora_sd[key], 'shape') else 'scalar'
            print(f"  [{i}] {key}: {shape}")
        
        # Check sizes
        total_params = sum(v.numel() for v in lora_sd.values() if hasattr(v, 'numel'))
        print(f"\nTotal parameters: {total_params:,}")
        
        file_size = Path(lora_path).stat().st_size / (1024 * 1024)
        print(f"File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"✗ Failed to inspect file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert ComfyUI Wan LoRAs to diffusers format')
    parser.add_argument('--input', '-i', required=True, help='Input ComfyUI LoRA file')
    parser.add_argument('--output', '-o', help='Output diffusers LoRA file (auto-generated if not provided)')
    parser.add_argument('--inspect', action='store_true', help='Inspect the input file without converting')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed conversion log')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        return
    
    if args.inspect:
        inspect_lora(input_path)
        return
    
    # Auto-generate output path if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_diffusers{input_path.suffix}"
    
    # Convert
    success = convert_comfy_wan_lora_to_diffusers(
        str(input_path),
        str(output_path),
        verbose=args.verbose
    )
    
    if success:
        print(f"\n✓ Conversion complete!")
        print(f"\nYou can now use this LoRA in diffusers:")
        print(f"  pipe.load_lora_weights('{output_path}', adapter_name='converted_lora')")
    else:
        print(f"\n✗ Conversion failed")


if __name__ == "__main__":
    main()