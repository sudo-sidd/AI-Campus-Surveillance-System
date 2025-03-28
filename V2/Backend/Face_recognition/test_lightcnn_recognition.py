import argparse
import cv2
import os
import sys
import time
import numpy as np
from face_recognize_lcnn import process_faces

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test LightCNN recognition on images, videos, or webcam")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "-i", help="Path to input image or video file")
    input_group.add_argument("--camera", "-c", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--output", "-o", help="Path to output image or video file (optional for webcam)")
    parser.add_argument("--threshold", "-t", type=float, default=0.45, help="Recognition threshold (default: 0.45)")
    return parser.parse_args()

def process_image(image_path, output_path, threshold):
    """Process a single image and save the annotated result"""
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return False
    
    # Process the image - without track_id for single images
    label, score, bboxes = process_faces(frame)
    
    # Annotate the image
    if bboxes[0]:
        x1, y1, x2, y2 = bboxes[0]
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with score
        if score >= threshold:
            text = f"{label} ({score:.2f})"
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save the annotated image
    cv2.imwrite(output_path, frame)
    print(f"Processed image saved to {output_path}")
    return True

def process_video(video_path, output_path, threshold):
    """Process a video and save the annotated result"""
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    processing_times = []
    track_id = 1  # Simulate a simple tracking ID
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 3rd frame for speed
        if frame_count % 3 == 0:
            start_time = time.time()
            
            # Process the frame with track_id for temporal consistency
            label, score, bboxes = process_faces(frame, track_id)
            
            # Annotate the frame
            if bboxes[0]:
                x1, y1, x2, y2 = bboxes[0]
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with score
                if score >= threshold:
                    text = f"{label} ({score:.2f})"
                    cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            end_time = time.time()
            processing_times.append(end_time - start_time)
            
            # Display processing speed
            fps_text = f"Processing: {1.0/(end_time - start_time):.1f} FPS"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Write the frame
        out.write(frame)
        frame_count += 1
        
        # Show progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
    
    # Print performance statistics
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"Average processing time per frame: {avg_time:.3f}s ({1.0/avg_time:.1f} FPS)")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")
    return True

def process_webcam(camera_index, output_path, threshold):
    """Process webcam stream with live display"""
    # Open webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return False
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create video writer if output is specified
    out = None
    if output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30  # Approximate FPS for webcam
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    processing_times = []
    track_id = 1  # Simulate a simple tracking ID
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from webcam")
            break
        
        # Process every 3rd frame for smoother real-time performance
        process_this_frame = frame_count % 3 == 0
        
        if process_this_frame:
            start_time = time.time()
            
            # Process the frame with track_id for temporal consistency
            label, score, bboxes = process_faces(frame, track_id)
            
            # Annotate the frame
            if bboxes[0]:
                x1, y1, x2, y2 = bboxes[0]
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with score
                if score >= threshold:
                    text = f"{label} ({score:.2f})"
                    cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            end_time = time.time()
            processing_times.append(end_time - start_time)
            
            # Display processing speed
            fps_text = f"Processing: {1.0/(end_time - start_time):.1f} FPS"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Write to output video if specified
        if out:
            out.write(frame)
        
        # Display the frame
        cv2.imshow("LightCNN Recognition", frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    # Print performance statistics
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"Average processing time per frame: {avg_time:.3f}s ({1.0/avg_time:.1f} FPS)")
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    if output_path:
        print(f"Webcam recording saved to {output_path}")
    
    return True

def main():
    args = parse_arguments()
    
    # Process based on input type
    if args.camera is not None:
        process_webcam(args.camera, args.output, args.threshold)
    else:
        # Check if input file exists
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist")
            return
        
        # Check if output is specified
        if not args.output:
            print("Error: Output file must be specified for image/video processing")
            return
            
        # Determine if input is image or video
        ext = os.path.splitext(args.input)[1].lower()
        is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp']
        
        if is_image:
            process_image(args.input, args.output, args.threshold)
        else:
            process_video(args.input, args.output, args.threshold)

if __name__ == "__main__":
    main()