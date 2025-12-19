import cv2
import os


def get_unique_filename(filepath):
    """
    Returns a unique filename by appending a number if the file already exists.
    """
    if not os.path.exists(filepath):
        return filepath
    
    base, ext = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1
    return f"{base}_{counter}{ext}"


def combine_videos(video_list, output_file_name):
    """
    Combines a list of MP4 video files into a single output file using OpenCV.
    
    Note: This method does not preserve audio.

    Args:
        video_list: A list of strings, where each string is the file path to an input video.
        output_file_name: The desired name for the combined output file (e.g., "final_video.mp4").
    """
    # Get properties from the first video
    first_video = cv2.VideoCapture(video_list[0])
    fps = first_video.get(cv2.CAP_PROP_FPS)
    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_video.release()

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file_name, fourcc, fps, (width, height))

    # Process each video
    for video_path in video_list:
        print(f"Processing: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()

    out.release()
    print(f"Successfully combined videos into {output_file_name}")


# --- Example Usage ---
if __name__ == "__main__":
    # List the paths to your input video files in the desired order
    input_files = [
        r"C:\Users\jared\Documents\ComfyUI\output\v2_1.mp4",
        r"C:\Users\jared\Documents\ComfyUI\output\v2_2.mp4",
        r"C:\Users\jared\Documents\ComfyUI\output\v2_3.mp4"
    ]
    output_name = f"combined_clips({len(input_files)}).mp4"
    
    # Output to the same directory as the first input file
    output_dir = os.path.dirname(input_files[0])
    output_path = os.path.join(output_dir, output_name)
    output_path = get_unique_filename(output_path)

    # Call the function to combine the videos
    combine_videos(input_files, output_path)
