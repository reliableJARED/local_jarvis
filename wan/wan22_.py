
import os
from huggingface_hub import snapshot_download
import torch
import gc
from diffusers import WanImageToVideoPipeline, FlowMatchEulerDiscreteScheduler,WanTransformer3DModel
from diffusers.utils import load_image, export_to_video

from transformers import T5EncoderModel, BitsAndBytesConfig

def download_wan22():
    base_dir = "./models"
    os.makedirs(base_dir, exist_ok=True)

    print(f"--- Downloading Wan 2.2 TI2V 5B ---")
    
    # Allowed *.bin files so Text Encoder downloads correctly
    model_path = snapshot_download(
        repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers", 
        local_dir=os.path.join(base_dir, "Wan2.2-TI2V-5B"),
        ignore_patterns=["*.msgpack"] 
    )
    print(f"✅ Model downloaded to: {model_path}")

class WAN22:
    def __init__(self, model_root="./models"):
        self.device = "cuda"
        self.model_path = os.path.join(model_root, "Wan2.2-TI2V-5B")
        self.pipeline = None
        
        print(">> Initializing Wan 2.2 [5B] on 5060 Ti...")
        self._load_pipeline()

    def _load_pipeline(self):
        # Config for 8-bit loading (Saves VRAM)
        # We use this for BOTH Text Encoder and Transformer to fit on 16GB
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["proj_out", "norm_out", "bias", "norm", "conv"] # Skip sensitive layers
        )
        
        print("... Loading Text Encoder (8-bit)")
        text_encoder = T5EncoderModel.from_pretrained(
            self.model_path,
            subfolder="text_encoder",
            quantization_config=quant_config,
            torch_dtype=torch.float16
        )

        #Load Transformer explicitly in 8-bit to prevent OOM
        print("... Loading Transformer (8-bit)")
        transformer = WanTransformer3DModel.from_pretrained(
            self.model_path,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.float16
        )

        #  Load Main Pipeline
        self.pipeline = WanImageToVideoPipeline.from_pretrained(
            self.model_path,
            text_encoder=text_encoder,
            transformer=transformer, 
            torch_dtype=torch.float16,
            #variant="fp16"
        )

        # 4. Offloading (Essential for 16GB)
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_vae_tiling()
        
        # 5. Set Scheduler
        self.pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )

    def generate(self, image, prompt, steps=25, seed=None):
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        print(f"--- Generating Video | Steps: {steps} | Seed: {seed} ---")

        with torch.no_grad():
            video = self.pipeline(
                image=image,
                prompt=prompt,
                negative_prompt="low quality, jittery, distorted, morphing, watermark, blur",
                height=720,
                width=1280,
                num_frames=81,       
                num_inference_steps=steps,
                guidance_scale=6.0, 
                generator=generator,
            ).frames
            
        return video

if __name__ == "__main__":
    

    download_wan22()
    
    bot = WAN22(model_root="./models")
    
    img_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    image = load_image(img_path)
    
    # 5B model usually needs ~25-30 steps for convergence without LoRA
    video_frames = bot.generate(
        image=image, 
        prompt="Cinematic slow motion, astronaut walking on mars, high detail, 4k, photorealistic",
        steps=30 
    )
    
    output_filename = "wan22_5b_result.mp4"
    export_to_video(video_frames, output_filename, fps=24)
    print(f"Done! Video saved to {output_filename}")