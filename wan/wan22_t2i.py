import torch
import subprocess
import os
from pathlib import Path
import sys
from huggingface_hub import snapshot_download


class Wan22TI2V:
    """
    Wan 2.2 Text-Image-to-Video Generator (TI2V-5B)
    Generates videos from text prompts or text+image using the Wan2.2-TI2V-5B model.
    
    This model uses the native Wan2.2 repository with generate.py script.
    It supports:
    - Text-to-Video (T2V) generation at 720P
    - Image-to-Video (I2V) generation at 720P
    - High compression VAE (4x16x16)
    - Runs on single GPU (24GB+ VRAM like RTX 4090)
    """
    
    def __init__(self, 
                 model_dir: str = None,
                 wan_repo_dir: str = None,
                 output_dir: str = "./output",
                 offload_model: bool = True,
                 convert_dtype: bool = True,
                 t5_cpu: bool = True):
        """
        Initialize the TI2V-5B pipeline.
        
        Args:
            model_dir: Path to Wan2.2-TI2V-5B model directory
            wan_repo_dir: Path to cloned Wan2.2 repository (optional, will detect)
            output_dir: Directory to save generated videos
            offload_model: Enable model offloading to reduce VRAM usage
            convert_dtype: Convert model dtype for memory efficiency
            t5_cpu: Run T5 text encoder on CPU to save VRAM
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory optimization flags for 24GB VRAM (e.g., RTX 4090)
        self.offload_model = offload_model
        self.convert_dtype = convert_dtype
        self.t5_cpu = t5_cpu
        
        # Use Hugging Face's built-in cache management
        if model_dir is None:
            print("Downloading/loading model from Hugging Face Hub...")
            model_dir = snapshot_download(
                repo_id="Wan-AI/Wan2.2-TI2V-5B",
                repo_type="model"
            )
            print(f"✓ Model loaded from: {model_dir}")
        
        self.model_dir = Path(model_dir)
        
        # Detect Wan2.2 repository
        if wan_repo_dir is None:
            # Try common locations
            possible_paths = [
                Path.cwd() / "Wan2.2",
                Path.cwd().parent / "Wan2.2",
                Path(__file__).parent / "Wan2.2",
                Path(__file__).parent.parent / "Wan2.2",
            ]
            
            for path in possible_paths:
                if (path / "generate.py").exists():
                    wan_repo_dir = str(path)
                    print(f"✓ Found Wan2.2 repo at: {wan_repo_dir}")
                    break
            
            if wan_repo_dir is None:
                print("⚠ Wan2.2 repository not found!")
                print("Clone with: git clone https://github.com/Wan-Video/Wan2.2.git")
                print("Then install: cd Wan2.2 && pip install -r requirements.txt")
        
        self.wan_repo_dir = Path(wan_repo_dir) if wan_repo_dir else None
        
        print(f"\n{'='*60}")
        print(f"Wan 2.2 TI2V-5B Video Generator")
        print(f"{'='*60}")
        print(f"Model directory: {self.model_dir}")
        print(f"Wan2.2 repo: {self.wan_repo_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Memory optimizations: offload={offload_model}, dtype={convert_dtype}, t5_cpu={t5_cpu}")
        print(f"{'='*60}\n")
    
    def generate_video(self, 
                      prompt: str, 
                      image_path: str = None,
                      size: str = "1280*704",
                      num_inference_steps: int = 20,
                      guidance_scale: float = 1.0,
                      seed: int = None,
                      video_name: str = None) -> Path:
        """
        Generate a video from text prompt or text+image.
        
        Args:
            prompt: Text description for video generation
            image_path: Optional image path for I2V mode (if None, uses T2V mode)
            size: Video resolution as "width*height" (default: "1280*704" for 720P)
            num_inference_steps: Number of denoising steps (default: 20)
            guidance_scale: Guidance scale for generation (default: 1.0)
            seed: Random seed for reproducibility
            video_name: Optional custom filename (without extension)
        
        Returns:
            Path to the saved video
        """
        if self.wan_repo_dir is None or not self.wan_repo_dir.exists():
            raise RuntimeError("Wan2.2 repository not found. Cannot generate video.")
        
        if not self.model_dir.exists():
            raise RuntimeError(f"Model directory not found: {self.model_dir}")
        
        mode = "I2V" if image_path else "T2V"
        print(f"\n{'='*60}")
        print(f"Generating video ({mode} mode)")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        if image_path:
            print(f"Image: {image_path}")
        print(f"Resolution: {size}")
        print(f"Steps: {num_inference_steps}")
        print(f"{'='*60}\n")
        
        # Build command
        cmd = [
            sys.executable,
            str(self.wan_repo_dir / "generate.py"),
            "--task", "ti2v-5B",
            "--size", size,
            "--ckpt_dir", str(self.model_dir),
            "--prompt", prompt,
            "--num_inference_steps", str(num_inference_steps),
            "--guidance_scale", str(guidance_scale),
        ]
        
        # Add memory optimization flags
        if self.offload_model:
            cmd.append("--offload_model")
            cmd.append("True")
        if self.convert_dtype:
            cmd.append("--convert_model_dtype")
        if self.t5_cpu:
            cmd.append("--t5_cpu")
        
        # Add image if provided
        if image_path:
            cmd.extend(["--image", str(image_path)])
        
        # Add seed if provided
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        # Run generation
        print(f"Running: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.wan_repo_dir),
                capture_output=True,
                text=True,
                check=True
            )
            
            print(result.stdout)
            if result.stderr:
                print("Warnings/Errors:", result.stderr)
            
            # Find the generated video
            # Wan2.2 typically outputs to a results directory
            results_dir = self.wan_repo_dir / "results"
            if results_dir.exists():
                videos = sorted(results_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
                if videos:
                    latest_video = videos[0]
                    
                    # Move to output directory with custom name
                    if video_name is None:
                        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c == " ").replace(" ", "_")
                        video_name = f"video_{safe_prompt}"
                    
                    output_path = self.output_dir / f"{video_name}.mp4"
                    
                    # Copy the video
                    import shutil
                    shutil.copy2(latest_video, output_path)
                    
                    print(f"\n{'='*60}")
                    print(f"✓ Video saved to: {output_path}")
                    print(f"{'='*60}\n")
                    
                    return output_path
            
            print("⚠ Could not find generated video in results directory")
            return None
            
        except subprocess.CalledProcessError as e:
            print(f"Error generating video: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
    
    def run_interactive_loop(self):
        """Run an interactive loop for generating videos from user prompts."""
        print("\n" + "="*60)
        print("Wan 2.2 TI2V-5B Video Generator - Interactive Mode")
        print("="*60)
        print("Commands:")
        print("  - Enter a text prompt to generate a video (T2V mode)")
        print("  - Enter 'image:path/to/image.jpg|prompt text' for I2V mode")
        print("  - Type 'quit' or 'exit' to stop")
        print("="*60)
        print("Note: TI2V-5B generates 720P videos at 24fps")
        print("      Supports both Text-to-Video and Image-to-Video")
        print("="*60 + "\n")
        
        video_count = 0
        
        while True:
            try:
                user_input = input("Enter prompt (or 'quit'/'exit' to stop): ").strip()
                
                if not user_input:
                    print("Please enter a valid prompt.")
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Exiting...")
                    break
                
                # Parse input for I2V mode
                image_path = None
                prompt = user_input
                
                if user_input.startswith("image:"):
                    # Format: image:path/to/image.jpg|prompt text
                    parts = user_input[6:].split("|", 1)
                    if len(parts) == 2:
                        image_path = parts[0].strip()
                        prompt = parts[1].strip()
                    else:
                        print("Invalid format. Use: image:path/to/image.jpg|prompt text")
                        continue
                
                video_count += 1
                self.generate_video(
                    prompt=prompt,
                    image_path=image_path,
                    video_name=f"video_{video_count}"
                )
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"Error generating video: {e}")
                import traceback
                traceback.print_exc()
                continue


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Wan 2.2 TI2V-5B Video Generator")
    parser.add_argument("--model_dir", type=str, default=None, 
                       help="Path to Wan2.2-TI2V-5B model directory")
    parser.add_argument("--wan_repo_dir", type=str, default=None,
                       help="Path to cloned Wan2.2 repository")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Directory to save generated videos")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Text prompt for video generation")
    parser.add_argument("--image", type=str, default=None,
                       help="Image path for I2V mode")
    parser.add_argument("--size", type=str, default="1280*704",
                       help="Video resolution (default: 1280*704 for 720P)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize the generator
    generator = Wan22TI2V(
        model_dir=args.model_dir,
        wan_repo_dir=args.wan_repo_dir,
        output_dir=args.output_dir
    )
    
    if args.interactive or args.prompt is None:
        # Run interactive loop
        generator.run_interactive_loop()
    else:
        # Single generation
        generator.generate_video(
            prompt=args.prompt,
            image_path=args.image,
            size=args.size
        )