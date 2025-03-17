import cv2
import time
from Person_detection import track_persons

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Testing person detection and pose estimation...")
    print("Press 'q' to quit")

    # For FPS calculation
    prev_time = time.time()
    fps = 0

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame from webcam")
            break

        # Process frame and calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Process frame
        try:
            results = track_persons(frame)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
        
        # Draw annotations
        for bbox, track_id in zip(results["person_boxes"], results["track_ids"]):
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add text
            text = f"ID: {track_id} (Facing Camera)"
            cv2.putText(frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display information
        persons_count = len(results["person_boxes"])
        cv2.putText(frame, f"Persons facing camera: {persons_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Person Detection Test', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()