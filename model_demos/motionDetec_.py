from dependency_manager import Orenda_DependencyManager
from typing import Tuple, TYPE_CHECKING
#ONLY for type hints, not actual imports - IDE friendly will set this to true, not at runtime
if TYPE_CHECKING:
    import numpy as np

class MotionDetector:
    def __init__(self, width: int = 640, height: int = 480, threshold: int = 10, min_contour_area: int = 100) -> None:
        """
        Initialize motion detector with configurable parameters. The algorithm is  conservative because it requires motion to be present
        in both frame differences (between frame1-frame2 AND frame2-frame3) using cv2.bitwise_and(). 
        This creates a more restrictive condition that only triggers on sustained, consistent motion.
        
        Args:
            width (int): Camera width
            height (int): Camera height
            threshold (int): Motion sensitivity threshold (lower = more sensitive)
            min_contour_area (int): Minimum area to consider as motion
        """
        # Import cv2 here since it should only be used after dependency check
        import cv2
        self.cv2 = cv2
        
        self.width = width
        self.height = height
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.cap = None
        self.frame1 = None
        self.frame2 = None
        
    def initialize_camera(self) -> bool:
        """Initialize the camera with specified dimensions."""
        self.cap = self.cv2.VideoCapture(0)
        self.cap.set(3, self.width)  # Set width
        self.cap.set(4, self.height)  # Set height
        
        #Discard first 10 frames
        for _ in range(10):  
            ret, _ = self.cap.read()
            if not ret:
                return False
        #now read first two frames to set motion detection context
        ret, self.frame1 = self.cap.read()
        ret, self.frame2 = self.cap.read()
        
        return ret
    
    def detect_motion(self, frame3: "np.ndarray") -> Tuple[bool, "np.ndarray"]:
        """
        Detect motion between three consecutive frames.
        
        Args:
            frame3: Current frame
            
        Returns:
            tuple: (motion_detected, processed_frame)
        """
        # Convert frames to grayscale
        gray1 = self.cv2.cvtColor(self.frame1, self.cv2.COLOR_BGR2GRAY)
        gray2 = self.cv2.cvtColor(self.frame2, self.cv2.COLOR_BGR2GRAY)
        gray3 = self.cv2.cvtColor(frame3, self.cv2.COLOR_BGR2GRAY)

        # Calculate frame differences
        diff1 = self.cv2.absdiff(gray1, gray2)
        diff2 = self.cv2.absdiff(gray2, gray3)

        # Find motion by combining differences
        motion = self.cv2.bitwise_and(diff1, diff2)
        
        # Apply threshold to get binary image
        _, motion_thresh = self.cv2.threshold(motion, self.threshold, 255, self.cv2.THRESH_BINARY)

        # Find contours of moving objects
        contours, _ = self.cv2.findContours(motion_thresh, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around detected motion
        motion_detected = False
        processed_frame = frame3.copy()
        
        for contour in contours:
            # Filter out small movements
            if self.cv2.contourArea(contour) > self.min_contour_area:  
                x, y, w, h = self.cv2.boundingRect(contour)
                self.cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                motion_detected = True
        
        return motion_detected, processed_frame
    
    def update_frames(self, new_frame: "np.ndarray") -> None:
        """Update frame sequence for next detection."""
        self.frame1 = self.frame2
        self.frame2 = new_frame
    
    def run(self) -> None:
        """Main detection loop."""
        if not self.initialize_camera():
            print("Error: Could not initialize camera")
            return
        
        print("Starting motion detection. Press 'q' to quit.")
        count = 0
        while True:
            ret, frame3 = self.cap.read()
            if not ret:
                break

            # Detect motion
            motion_detected, processed_frame = self.detect_motion(frame3)
            
            # Print when motion is detected
            if motion_detected:
                count += 1
                print(f"Motion detected for the {count} time")
                
            # Show the results
            self.cv2.imshow('Motion Detection', processed_frame)

            # Update frames for next iteration
            self.update_frames(frame3)

            # Exit on 'q' key press
            if self.cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        self.cleanup()
    
    def cleanup(self) -> None:
        """Release resources."""
        if self.cap:
            self.cap.release()
        self.cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    dep_manager = Orenda_DependencyManager()
    if dep_manager.run(download_models=False):
        print("All dependencies are ready to use!")

        # Create and run motion detector
        detector = MotionDetector()
        detector.run()
    else:
        print("Dependencies not ready. Cannot start motion detection.")