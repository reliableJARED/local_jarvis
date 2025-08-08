"""
Test script for Wan2.2 Image-to-Video with memory optimization
"""

import os
import sys
import torch
from wan22_video_i2v import Wan22ImageToVideo

def test_memory_optimization():
    """Test the memory optimization features"""
    print("Testing Wan2.2 Image-to-Video Memory Optimization")
    print("=" * 50)
    
    # Initialize generator with maximum memory optimization
    generator = Wan22ImageToVideo(optimize_for_low_memory=True)
    
    # Print detailed system information
    print("\nDetailed System Information:")
    system_info = generator.get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Check dependencies
    print("\nDependency Status:")
    deps = generator.check_dependencies()
    missing_deps = []
    for dep, status in deps.items():
        status_symbol = '✓' if status else '✗'
        print(f"  {dep}: {status_symbol}")
        if not status and dep not in ['flash_attn']:  # flash_attn is optional
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\nMissing critical dependencies: {missing_deps}")
        print("Please install them before proceeding.")
        return False
    
    # Test GPU memory check
    print(f"\nGPU Memory Check:")
    if torch.cuda.is_available():
        memory_ok = generator.check_gpu_memory_available(2.0)
        print(f"  Sufficient memory available: {memory_ok}")
        
        # Show optimization recommendations
        optimized_settings = generator.optimize_generation_settings("1280*704", 5)
        print(f"  Recommended settings: {optimized_settings}")
    
    # Test with the actual image if it exists
    image_path = r"C:\Users\jared\Documents\code\local_jarvis\xserver\demetra\zeus\demetra_in_zeus-p4_a4_f3_c2.png"
    
    if os.path.exists(image_path):
        print(f"\nTesting with image: {image_path}")
        prompt = "the woman gets up from the chair and walks towards the camera"
        
        # Use very conservative settings to avoid memory issues
        print("Using ultra-conservative memory settings:")
        test_settings = {
            'duration': 1,  # Minimal duration
            'fps': 12,      # Lower FPS
            'size': "256*256",  # Very small resolution  
            'guidance_scale': 4.0,  # Minimal guidance
            'num_inference_steps': 10  # Minimal steps
        }
        print(f"  Settings: {test_settings}")
        
        try:
            # Start memory monitoring
            monitor_thread = generator.monitor_memory_during_generation(log_interval=5)
            
            output_video = generator.generate_video(
                image_path=image_path,
                prompt=prompt,
                **test_settings
            )
            
            # Stop memory monitoring
            generator.stop_memory_monitoring()
            
            if output_video:
                print(f"\n✓ Test successful! Video generated: {output_video}")
                return True
            else:
                print(f"\n✗ Test failed: Video generation failed")
                return False
                
        except Exception as e:
            generator.stop_memory_monitoring()
            print(f"\n✗ Test failed with exception: {e}")
            return False
    else:
        print(f"\nTest image not found: {image_path}")
        print("Please update the image path in the test script.")
        return False

if __name__ == "__main__":
    success = test_memory_optimization()
    if success:
        print("\n🎉 Memory optimization test completed successfully!")
    else:
        print("\n❌ Memory optimization test failed. Check the logs above.")
