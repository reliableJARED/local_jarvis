from typing import Dict, List, Tuple, Any
import numpy as np
from dependency_manager import Orenda_DependencyManager

class YOLOhf:
    def __init__(self, model_name: str = "hustvl/yolos-tiny") -> None:
        """
        Initialize the Hugging Face YOLOS model using the official demo approach
        
        Args:
            model_name: The Hugging Face model identifier
        """
        import cv2
        from PIL import Image
        import torch
        from transformers import YolosImageProcessor, YolosForObjectDetection
        
        self.cv2 = cv2
        self.torch = torch
        self.Image = Image

        print(f"Loading model: {model_name}")
        self.model: YolosForObjectDetection = YolosForObjectDetection.from_pretrained(model_name)
        self.image_processor: YolosImageProcessor = YolosImageProcessor.from_pretrained(model_name)
        
        # Get class names from model config
        self.class_names: Dict[int, str] = self.model.config.id2label
        print(f"Loaded model with {len(self.class_names)} classes")
        
    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Perform object detection on a frame using the HF demo approach
        
        Args:
            frame: Input image as numpy array (BGR format from OpenCV)
            confidence_threshold: Minimum confidence score for detections
            
        Returns:
            Dictionary containing detection results with 'scores', 'labels', and 'boxes'
        """
        # Convert OpenCV BGR to RGB
        rgb_frame: np.ndarray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        pil_image = self.Image.fromarray(rgb_frame)
        
        # Process the image
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs using the same approach as HF demo
        target_sizes = self.torch.tensor([pil_image.size[::-1]])  # (height, width)
        results: List[Dict[str, Any]] = self.image_processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes
        )
        
        return results[0]
    
    def draw_detections(self, frame: np.ndarray, results: Dict[str, Any], 
                       filter_person_only: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: Input image as numpy array
            results: Detection results from detect() method
            filter_person_only: Whether to only draw person detections
            
        Returns:
            Annotated frame as numpy array
        """
        annotated_frame: np.ndarray = frame.copy()
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # Filter for person class only if specified (person is typically class 0)
            class_name: str = self.class_names[label.item()]
            if filter_person_only and class_name.lower() != 'person':
                continue
                
            box_coords: List[int] = [round(i) for i in box.tolist()]
            x1, y1, x2, y2 = box_coords
            
            # Draw bounding box
            self.cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label and confidence
            confidence: float = score.item()
            label_text: str = f"{class_name}: {confidence:.2f}"
            
            # Calculate text size and draw background rectangle
            text_size: Tuple[Tuple[int, int], int] = self.cv2.getTextSize(
                label_text, self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            (text_width, text_height), baseline = text_size
            
            self.cv2.rectangle(
                annotated_frame, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                (0, 255, 0), 
                -1
            )
            
            # Draw text
            self.cv2.putText(
                annotated_frame, 
                label_text, 
                (x1, y1 - 5), 
                self.cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                1
            )
        
        return annotated_frame

def main() -> None:
    """Main function to run the person detection system"""
    dep_manager: Orenda_DependencyManager = Orenda_DependencyManager()
    if dep_manager.run(download_models=False):
        print("All dependencies are ready to use!")

    # Initialize the model
    print("Initializing Hugging Face YOLOS model...")
    yolo_model: YOLOhf = YOLOhf()

    print("Available classes:")
    for idx, class_name in yolo_model.class_names.items():
        print(f"{idx}: {class_name}")

    # Open the video stream
    cap = yolo_model.cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height

    print("Starting person detection. Press 'q' to quit.")

    while True:
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame, exiting...")
            break
        
        # Perform detection
        try:
            results: Dict[str, Any] = yolo_model.detect(frame, confidence_threshold=0.5)
            
            # Print detections (similar to HF demo)
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                class_name: str = yolo_model.class_names[label.item()]
                if class_name.lower() == 'person':  # Only print person detections
                    box_rounded: List[float] = [round(i, 2) for i in box.tolist()]
                    print(f"Detected {class_name} with confidence {round(score.item(), 3)} at location {box_rounded}")
            
            # Draw annotations (filter for person only)
            annotated_frame: np.ndarray = yolo_model.draw_detections(frame, results, filter_person_only=True)
            
            # Display the frame
            yolo_model.cv2.imshow('Person Detection - Hugging Face YOLOS', annotated_frame)
            
        except Exception as e:
            print(f"Detection error: {e}")
            yolo_model.cv2.imshow('Person Detection - Hugging Face YOLOS', frame)
        
        # Press 'q' to quit
        if yolo_model.cv2.waitKey(1) == ord('q'):
            break

    # Cleanup
    cap.release()
    yolo_model.cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    main()