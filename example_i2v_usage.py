"""
Example usage of Wan2.2 5B Image-to-Video Generator
Demonstrates how to use the modular wrapper for video generation

Author: Jared
Date: August 2025
"""

import os
import sys
from pathlib import Path
from wan.wan22_video_i2v import Wan22ImageToVideo, generate_video_from_image

def example_basic_usage():
    """Basic example of image-to-video generation"""
    print("=== Basic Image-to-Video Generation Example ===")
    
    # Initialize the generator
    generator = Wan22ImageToVideo()
    
    # Print system info
    print("\nSystem Information:")
    system_info = generator.get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Check if model exists
    if not system_info['model_exists']:
        print("\n❌ Model not found! Please run setup first:")
        print("  python setup_wan22.py --all")
        return False
    
    # Example image path
    #example_image = r"C:\Users\jared\Documents\code\local_jarvis\xserver\demetra\zeus\demetra_in_zeus-p4_a4_f3_c2.png"
    example_image = "./xserver/demetra/zeus/demetra_in_zeus-p4_a4_f3_c2.png"
    
    if not os.path.exists(example_image):
        print(f"\n❌ Example image not found: {example_image}")
        print("Please provide an image file or update the path")
        return False
    
    # Generate video with a specific prompt
    prompt = "The woman gets up from chair and walks towards camera"
    
    print(f"\n🎬 Generating video from: {example_image}")
    print(f"📝 Prompt: {prompt}")
    
    output_path = generator.generate_video(
        image_path=example_image,
        prompt=prompt,
        size="1280*704",  # 720p optimized for 5B model
    )
    
    if output_path and os.path.exists(output_path):
        print(f"✅ Video generated successfully: {output_path}")
        return True
    else:
        print("❌ Video generation failed")
        return False

def example_advanced_usage():
    """Advanced example with custom parameters"""
    print("\n=== Advanced Image-to-Video Generation Example ===")
    
    # Initialize with custom settings
    model_path = os.path.join(os.getcwd(), "Wan2.2-TI2V-5B")
    generator = Wan22ImageToVideo(
        model_path=model_path,
        optimize_for_low_memory=True  # Essential for 16GB GPU
    )
    
    # Example parameters
    example_image = "./xserver/demetra/zeus/demetra_in_zeus-p4_a4_f3_c2.png"
    
    if not os.path.exists(example_image):
        print(f"❌ Example image not found: {example_image}")
        return False
    
    # Advanced generation with custom parameters
    output_path = generator.generate_video(
        image_path=example_image,
        prompt="Cinematic slow motion, dramatic lighting changes, smooth camera movement",
        output_path="advanced_output.mp4",
        size="1280*704",
        seed=42,  # For reproducible results
        guidance_scale=7.5,
        num_inference_steps=50
    )
    
    if output_path and os.path.exists(output_path):
        print(f"✅ Advanced video generated: {output_path}")
        
        # Cleanup GPU memory
        generator.cleanup()
        return True
    else:
        print("❌ Advanced video generation failed")
        return False

def example_batch_processing():
    """Example of processing multiple images"""
    print("\n=== Batch Processing Example ===")
    
    generator = Wan22ImageToVideo()
    
    # Example image list (update with your actual images)
    image_list = [
        {"path": "image1.jpg", "prompt": "Gentle wind blowing through leaves"},
        {"path": "image2.jpg", "prompt": "Ocean waves softly moving"},
        {"path": "image3.jpg", "prompt": "Clouds drifting across the sky"},
    ]
    
    results = []
    
    for i, item in enumerate(image_list):
        if not os.path.exists(item["path"]):
            print(f"⚠️  Skipping {item['path']} - file not found")
            continue
        
        print(f"\n🎬 Processing {i+1}/{len(image_list)}: {item['path']}")
        
        output_path = generator.generate_video(
            image_path=item["path"],
            prompt=item["prompt"],
            output_path=f"batch_output_{i+1}.mp4"
        )
        
        if output_path:
            results.append(output_path)
            print(f"✅ Generated: {output_path}")
        else:
            print(f"❌ Failed to generate video for {item['path']}")
        
        # Clear GPU memory between generations
        generator.cleanup()
    
    print(f"\n🎉 Batch processing completed. Generated {len(results)} videos.")
    return results

def example_quick_function():
    """Example using the convenience function"""
    print("\n=== Quick Function Example ===")
    
    example_image = "./xserver/demetra/zeus/demetra_in_zeus-p4_a4_f3_c2.png"
    
    if not os.path.exists(example_image):
        print(f"❌ Example image not found: {example_image}")
        return False
    
    # Use the quick convenience function
    output_path = generate_video_from_image(
        image_path=example_image,
        prompt="Beautiful cinematic movement with natural lighting",
        output_path="quick_output.mp4"
    )
    
    if output_path:
        print(f"✅ Quick generation completed: {output_path}")
        return True
    else:
        print("❌ Quick generation failed")
        return False

def check_setup():
    """Check if everything is properly set up"""
    print("=== Setup Check ===")
    
    generator = Wan22ImageToVideo()
    
    # Check dependencies
    print("\nDependency Status:")
    deps = generator.check_dependencies()
    all_deps_ok = True
    
    for dep, status in deps.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {dep}")
        if not status and dep != 'flash_attn':  # flash_attn is optional
            all_deps_ok = False
    
    # Check system info
    print("\nSystem Information:")
    system_info = generator.get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Overall status
    model_exists = system_info.get('model_exists', False)
    cuda_available = system_info.get('cuda_available', False)
    
    print(f"\n📊 Setup Status:")
    print(f"  Dependencies: {'✅' if all_deps_ok else '❌'}")
    print(f"  Model: {'✅' if model_exists else '❌'}")
    print(f"  CUDA: {'✅' if cuda_available else '⚠️'}")
    
    if not model_exists:
        print(f"\n💡 To download the model, run:")
        print(f"  python setup_wan22.py --download-model --clone-repo")
    
    if not all_deps_ok:
        print(f"\n💡 To install dependencies, run:")
        print(f"  python setup_wan22.py --install-deps")
    
    return all_deps_ok and model_exists

def main():
    """Main function to run examples"""
    print("🎬 Wan2.2 5B Image-to-Video Generator Examples")
    print("=" * 50)
    
    # First check setup
    if not check_setup():
        print("\n❌ Setup incomplete. Please run setup first:")
        print("  python setup_wan22.py --all")
        return
    
    print("\n🚀 Setup looks good! Running examples...")
    
    # Run examples
    try:
        # Basic example
        example_basic_usage()
        
        # Advanced example  
        example_advanced_usage()
        
        # Quick function example
        example_quick_function()
        
        # Batch processing example (commented out by default)
        # example_batch_processing()
        
        print("\n🎉 All examples completed!")
        print("\n💡 Tips for 16GB GPU:")
        print("  - Use 1280*704 resolution for best performance")
        print("  - Enable memory optimizations (default)")
        print("  - Process one video at a time")
        print("  - Clear GPU cache between generations")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
