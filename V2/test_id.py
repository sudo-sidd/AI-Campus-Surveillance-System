import cv2
import numpy as np
import os
from Backend.Person_detection.Person_detection import track_persons
from Backend.ID_detection.yolov11.ID_Detection import detect_id_card

def preprocess_frame(frame):
    """Enhances contrast using CLAHE in LAB color space."""
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_channel)

        # Merge channels and convert back to BGR
        merged_lab = cv2.merge((l_clahe, a_channel, b_channel))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return frame

def draw_annotations(frame, person_data):
    """Draw bounding boxes and annotations on the frame."""
    try:
        for person in person_data:
            # Draw person bounding box
            x1, y1, x2, y2 = person['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw ID card box if detected
            if person['id_flag']:
                ib_x1, ib_y1, ib_x2, ib_y2 = person['id_box']
                cv2.rectangle(frame, (ib_x1, ib_y1), (ib_x2, ib_y2), (255, 0, 0), 2)
                
                # Prepare text annotations
                text = f"ID: {person['track_id']} | IDCard: {person['id_card']}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Add background and text
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), (255, 255, 255), -1)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:
                # Prepare text annotations for person without ID
                text = f"ID: {person['track_id']} | No ID Card"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Add background and text
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), (255, 255, 255), -1)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    except Exception as e:
        print(f"Error in draw_annotations: {e}")
    
    return frame

def process_video(video_path, output_path=None, process_every_n_frames=5, show_output=True):
    """Process a video file, detect persons and ID cards, and return annotated frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Calculate output FPS based on processing rate
    output_fps = fps / process_every_n_frames
    print(f"Output FPS: {output_fps:.2f} (every {process_every_n_frames} frames)")
    
    # Initialize video writer if output path is specified
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    else:
        writer = None
    
    if show_output:
        cv2.namedWindow('Processed Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Processed Video', 1280, 720)
    
    frame_count = 0
    processed_count = 0
    
    while True:
        ret, original_frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames that we're not processing
        if frame_count % process_every_n_frames != 0:
            continue
            
        # We're now only dealing with frames we want to process
        processed_count += 1
        
        # Copy the original frame for annotation
        annotated_frame = original_frame.copy()
        
        # Person detection on original frame
        person_results = track_persons(original_frame)
        
        # Check if we have valid detection results
        if person_results and "person_boxes" in person_results and "track_ids" in person_results:
            person_boxes = person_results["person_boxes"]
            track_ids = person_results["track_ids"]
            people_data = []
            
            # Process each detected person
            for person_box, track_id in zip(np.array(person_boxes).tolist(), track_ids):
                try:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = [int(coord) for coord in person_box]
                    
                    # Validate and clip bounding boxes against the original frame
                    frame_height, frame_width, _ = original_frame.shape
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)
                    
                    # Crop the person from the original high-resolution frame
                    high_res_person_image = original_frame[y1:y2, x1:x2]
                    if high_res_person_image.size == 0:
                        print(f"Empty image for track_id: {track_id}")
                        continue
                    
                    # Initialize person data
                    person = {
                        'bbox': [x1, y1, x2, y2],
                        'track_id': track_id,
                        'id_flag': False,
                        'id_card': 'none',
                        'id_box': [0, 0, 0, 0]
                    }
                    
                    # Apply preprocessing while maintaining resolution
                    enhanced_person_image = preprocess_frame(high_res_person_image)
                    
                    # ID card detection on the high-resolution crop
                    try:
                        id_flag, id_box, id_card = detect_id_card(enhanced_person_image)
                        person['id_flag'] = id_flag
                        person['id_card'] = id_card
                        
                        if id_flag and id_box:
                            ib_x1, ib_y1, ib_x2, ib_y2 = id_box
                            # Map ID box coordinates to original frame coordinates
                            ib_x1 += x1
                            ib_y1 += y1
                            ib_x2 += x1
                            ib_y2 += y1
                            person['id_box'] = [ib_x1, ib_y1, ib_x2, ib_y2]
                    except Exception as e:
                        print(f"ID card detection error: {e}")
                    
                    # Add person to the data list
                    people_data.append(person)
                    
                except Exception as e:
                    print(f"Error processing person: {e}")
                    continue
            
            # Draw annotations on the copy of the original frame
            annotated_frame = draw_annotations(annotated_frame, people_data)
            
            print(f"Processed frame {frame_count}, found {len(people_data)} people")
        else:
            print(f"No persons detected in frame {frame_count}")
        
        # Write the annotated frame
        if writer:
            writer.write(annotated_frame)
        
        # Display the annotated frame
        if show_output:
            cv2.imshow('Processed Video', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if show_output:
        cv2.destroyAllWindows()
    
    print(f"Video processing completed. Processed {processed_count} of {frame_count} frames.")
    return True

def main():
    # Get input video path from command line or use default
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "/mnt/data/PROJECTS/Face_rec-ID_detection/vid/new_test2.mp4"
    
    # Set output path
    output_dir = os.path.dirname(video_path)
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_processed{ext}")
    
    # Process the video
    process_every_n_frames = 5  # Process every 5th frame
    process_video(video_path, output_path, process_every_n_frames)

if __name__ == "__main__":
    main()