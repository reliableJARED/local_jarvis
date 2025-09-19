import sys
from typing import Optional, Generator, Dict, List, Any, Union, Tuple, TYPE_CHECKING
from types import TracebackType


#ONLY for type hints, not actual imports - IDE friendly will set this to true, not at runtime
if TYPE_CHECKING:
    import numpy as np
    from PIL import Image
    import cv2
    import torch
    from transformers import AutoModelForCausalLM


class CameraHandler:
    """Simple OpenCV camera handler for video streaming and single image capture"""

    def __init__(self, camera_index: int = 0) -> None:
        """Initialize camera with specified index"""
        from PIL import Image
        import cv2

        self.camera_index: int = camera_index
        self.camera: Optional[cv2.VideoCapture] = None
        self.cv2 = cv2
        self.Image = Image
        self._initialize_camera()
    
    def _initialize_camera(self) -> None:
        """Initialize the camera connection"""
        self.camera = self.cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")
    
    def get_frame(self) -> "np.ndarray":
        """Get a single frame from camera (returns OpenCV BGR format)"""
        if not self.camera or not self.camera.isOpened():
            raise RuntimeError("Camera not initialized or closed")
        
        ret: bool
        frame: "np.ndarray"
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")
        
        return frame
    
    def get_frame_as_pil(self, warmup_frames: int = 10) -> "Image.Image":
        """Get a single frame as PIL Image (RGB format) with optional warmup"""
        # Warmup camera by discarding frames
        for _ in range(warmup_frames):
            ret: bool
            _: "np.ndarray"
            ret, _ = self.camera.read()
            if not ret:
                raise RuntimeError("Failed to capture frame during warmup")
        
        # Capture actual frame
        frame: "np.ndarray" = self.get_frame()
        
        # Convert BGR to RGB for PIL
        frame_rgb: "np.ndarray" = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        return self.Image.fromarray(frame_rgb)
    
    def stream_frames(self) -> Generator["np.ndarray", None, None]:
        """Generator that yields frames continuously"""
        while True:
            try:
                frame: "np.ndarray" = self.get_frame()
                yield frame
            except RuntimeError:
                break
    
    def release(self) -> None:
        """Release camera resources"""
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def __enter__(self) -> 'CameraHandler':
        return self
    
    def __exit__(
        self, 
        exc_type: Optional[type], 
        exc_val: Optional[BaseException], 
        exc_tb: Optional[TracebackType]
    ) -> None:
        self.release()


#TODO:
"""
- NOT for this model demo, but in general when streaming frames,
consider opencv color comparison or motion detection to only
process frames that have changed significantly (to save compute)
"""


class MoondreamWrapper:
    """Wrapper class for Moondream2 vision model operations"""
    
    def __init__(self,local_files_only:bool=False, model_name: str = "vikhyatk/moondream2", revision: str = "2025-01-09") -> None:
        """Initialize Moondream model with automatic device detection"""
        import numpy as np
        import cv2
        import torch
        from transformers import AutoModelForCausalLM

        self.cv2 = cv2
        self.torch = torch
        self.np = np
        self.AutoModelForCausalLM = AutoModelForCausalLM

        self.model_name: str = model_name
        self.revision: str = revision
        self.model: Optional["AutoModelForCausalLM"] = None
        self.device: Optional[str] = None
        self.dtype: Optional["torch.dtype"] = None
        
        self._setup_device()
        self._load_model(local_only=local_files_only)
        

    def _setup_device(self) -> None:
        """Detect and configure optimal device and dtype"""
        if self.torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = self.torch.float16
            print("Using Apple Silicon GPU (MPS)")
        elif self.torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = self.torch.float16
            print("Using NVIDIA GPU (CUDA)")
        else:
            self.device = "cpu"
            self.dtype = self.torch.float32
            print("Using CPU")
        
        print(f"Device: {self.device}, Data type: {self.dtype}")
    
    def _load_model(self,local_only:bool=False) -> None:
        """Load the Moondream model"""
        print("Loading Moondream2 model...")
        try:
            self.model = self.AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.revision,
                trust_remote_code=True,
                device_map={"": self.device},
                dtype=self.dtype,
                local_files_only=local_only
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            
            # Check for libvips error and provide help
            if "libvips" in str(e).lower():
                print("\nLIBVIPS ERROR DETECTED")
                print("Common solutions:")
                print("1. conda install -c conda-forge libvips")
                print("2. winget install libvips.libvips")
                print("3. Download from: https://libvips.github.io/libvips/install.html")
                print("Restart terminal after installation")
            
            raise RuntimeError(f"Model loading failed: {e}")
    
    def encode_image(self, image: Union["Image.Image", "np.ndarray"]) -> Any:
        """Encode image for multiple operations (more efficient)"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        return self.model.encode_image(image)
    
    def caption_image(
        self, 
        image: Union["Image.Image", "np.ndarray", Any], 
        length: str = "normal", 
        stream: bool = False
    ) -> str:
        """Generate image caption"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        result: Dict[str, Any] = self.model.caption(image, length=length, stream=stream)
        return result["caption"]
    
    def ask_question(self, image: Union["Image.Image", "np.ndarray", Any], question: str) -> str:
        """Perform visual question answering"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        result: Dict[str, Any] = self.model.query(image, question)
        return result["answer"]
    
    def detect_objects(self, image: Union["Image.Image", "np.ndarray", Any], object_type: str) -> List[Any]:
        """Detect specific objects in image"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        result: Dict[str, Any] = self.model.detect(image, object_type)
        return result["objects"]
    
    def point_to_objects(self, image: Union["Image.Image", "np.ndarray", Any], object_description: str) -> List[Tuple[int, int]]:
        """Get point coordinates for objects in image"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        result: Dict[str, Any] = self.model.point(image, object_description)
        return result["points"]
    
    def analyze_image_complete(self, image: Union["Image.Image", "np.ndarray"]) -> Dict[str, Any]:
        """Complete analysis of an image with all available methods"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Encode image once for efficiency
        encoded_image: Any = self.encode_image(image)
        
        results: Dict[str, Any] = {
            "short_caption": self.caption_image(encoded_image, length="short"),
            "detailed_caption": self.caption_image(encoded_image, length="normal"),
            "face_detection": self.detect_objects(encoded_image, "face"),
            "person_detection": self.detect_objects(encoded_image, "person"),
            "person_points": self.point_to_objects(encoded_image, "person")
        }
        
        return results


# Example usage
if __name__ == "__main__":

    # Initialize camera and model
    moondream: MoondreamWrapper = MoondreamWrapper()
    try:
        with CameraHandler(camera_index=0) as camera:
            
            # Capture single image
            print("Capturing image from camera...")
            image: "Image.Image" = camera.get_frame_as_pil(warmup_frames=10)
            print("Image captured successfully")
            
            # Analyze the image
            print("\nAnalyzing image...")
            
            # Short caption
            caption: str = moondream.caption_image(image, length="short")
            print(f"Caption: {caption}")
            
            # Ask questions
            questions: List[str] = [
                "Do you see a person in the image?",
                "What is the main subject of this image?",
                "What colors are most prominent?"
            ]
            
            for question in questions:
                answer: str = moondream.ask_question(image, question)
                print(f"Q: {question}")
                print(f"A: {answer}\n")
            
            # Detect objects
            faces: List[Any] = moondream.detect_objects(image, "face")
            people: List[Any] = moondream.detect_objects(image, "person")
            
            print(f"Found {len(faces)} face(s)")
            print(f"Found {len(people)} person(s)")
            
            # Complete analysis
            results: Dict[str, Any] = moondream.analyze_image_complete(image)
            print("\nComplete analysis results:")
            for key, value in results.items():
                print(f"{key}: {value}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    
    print("Single Image Demo completed successfully... STREAM Demo:")

    try:
        with CameraHandler(camera_index=0) as camera:
            #moondream: MoondreamWrapper = MoondreamWrapper()
            
            print("Starting continuous frame analysis... Press Ctrl+C to stop")
            print("Press 'q' in the camera window to quit")
            
            while True:
                # Capture single image (no warmup needed in continuous loop)
                image: "Image.Image" = camera.get_frame_as_pil(warmup_frames=0)
               
                # Short caption
                caption: str = moondream.caption_image(image, length="short")
                image_analysis = {"visual_scene": caption}
                print(f"\n {image_analysis}")
                
                # Convert PIL image back to OpenCV format for display
                frame_bgr: "np.ndarray" = moondream.cv2.cvtColor(
                    moondream.np.array(image), 
                    moondream.cv2.COLOR_RGB2BGR
                )
                
                # Add caption text overlay on the image
                moondream.cv2.putText(
                    frame_bgr,              # 1. Image to draw on
                    caption[:25],           # 2. Text string (truncated to 25 chars)
                    (10, 30),              # 3. Position (x, y) - bottom-left corner of text
                    moondream.cv2.FONT_HERSHEY_SIMPLEX, # 4. Font type
                    0.7,                   # 5. Font scale (size multiplier)
                    (0, 255, 0),          # 6. Color in BGR format (Blue, Green, Red)
                    2                      # 7. Thickness of text lines
                )
                
                # Show the frame with caption
                moondream.cv2.imshow('Camera Stream', frame_bgr)
                
                # Break loop if 'q' is pressed or window is closed
                key: int = moondream.cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("\nStream interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"Stream error: {e}")
    finally:
        print("Cleaning up stream resources...")
        moondream.cv2.destroyAllWindows()