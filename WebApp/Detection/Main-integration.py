import cv2
from aiohttp import web
from Face_recognition.face_recognize import recognize_face
from ID_detection.yolov11.ID_Detection import detect_id_card

async def video_feed(request):
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={'Content-Type': 'multipart/x-mixed-replace; boundary=frame'}
    )
    await response.prepare(request)

    cap = cv2.VideoCapture(0)  # Access the default camera
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Apply ID card detection processing
        modified_frame, bounding_boxes, associations = detect_id_card(frame)  # Ensure this returns correct values

        # Apply face recognition processing
        modified_frame = recognize_face(modified_frame)  # Apply face recognition on the processed frame


        _, buffer = cv2.imencode('.jpg', modified_frame)
        frame_bytes = buffer.tobytes()

        # Write the frame to the response
        await response.write(b'--frame\r\n'
                             b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    await response.write_eof()
    cap.release()
    return response

app = web.Application()
app.router.add_get('/video_feed', video_feed)

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=5000)
