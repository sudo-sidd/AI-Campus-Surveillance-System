# import cv2
# import time
# from Person_detection import track_persons

# def main():
#     cap = cv2.VideoCapture("V2/Backend/cut_crowd.mp4")
    
#     # Check if camera opened successfully
#     if not cap.isOpened():
#         print("Error: Could not open webcam")
#         return
    
#     print("Testing person detection and pose estimation...")
#     print("Press 'q' to quit")
    
#     # For FPS calculation
#     prev_time = time.time()
#     fps = 0
#     frame_count = 0
    
#     while True:
#         # Read frame from webcam
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Can't receive frame from webcam")
#             break
        
#         frame_count += 1
#         if frame_count % 5 == 0:
#             continue
        
#         # Process frame and calculate FPS
#         current_time = time.time()
#         fps = 1 / (current_time - prev_time)
#         prev_time = current_time
        
#         # Process frame
#         try:
#             results = track_persons(frame)
            
#             # Get height ratios (distance proxies) for each detection
#             height_ratios = []
#             frame_height = frame.shape[0]
            
#             for bbox in results["person_boxes"]:
#                 x1, y1, x2, y2 = [int(coord) for coord in bbox]
#                 height = y2 - y1
#                 height_ratio = height / frame_height
#                 height_ratios.append(height_ratio)
            
#         except Exception as e:
#             print(f"Error processing frame: {e}")
#             continue
        
#         # Display bounding boxes and distance info
#         for i, (bbox, track_id) in enumerate(zip(results["person_boxes"], results["track_ids"])):
#             x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
#             # Draw bounding box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
#             # Add distance info (using height ratio)
#             height_ratio = height_ratios[i]
#             distance_text = f"ID: {track_id}, Dist: {height_ratio:.2f}"
#             cv2.putText(frame, distance_text, 
#                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         persons_count = len(results["person_boxes"])
#         cv2.putText(frame, f"FPS: {fps:.1f}", 
#                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         cv2.putText(frame, f"People: {persons_count}", 
#                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
#         frame = cv2.resize(frame, (640*2, 480*2))
        
#         # Show frame
#         cv2.imshow('Person Detection Test', frame)
        
#         # Break loop on 'q' press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Clean up
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


import cv2
import time
from Person_detection import track_persons, draw_pose_landmarks

def main():
    cap = cv2.VideoCapture("V2/Backend/cut_crowd.mp4")
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    print("Testing person detection and pose estimation...")
    print("Press 'q' to quit")
    
    # For FPS calculation
    prev_time = time.time()
    fps = 0
    frame_count = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break
        
        frame_count += 1
        if frame_count % 5 == 0:
            continue
        
        # Process frame and calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Process frame
        try:
            # Set run_pose_estimation=True to enable pose estimation
            results = track_persons(frame, run_pose_estimation=True)
            
            # Draw pose landmarks
            frame_with_pose = draw_pose_landmarks(frame, results["pose_results"])
            
            # Draw bounding boxes and distance info
            for i, (bbox, track_id) in enumerate(zip(results["person_boxes"], results["track_ids"])):
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Draw bounding box
                cv2.rectangle(frame_with_pose, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add distance info
                height_ratio = results["height_ratios"][i]
                distance_text = f"ID: {track_id}, Dist: {height_ratio:.2f}"
                cv2.putText(frame_with_pose, distance_text, 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
        
        persons_count = len(results["person_boxes"])
        cv2.putText(frame_with_pose, f"FPS: {fps:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_pose, f"People: {persons_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        frame_with_pose = cv2.resize(frame_with_pose, (640*2, 480*2))
        
        # Show frame
        cv2.imshow('Person Detection Test', frame_with_pose)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()