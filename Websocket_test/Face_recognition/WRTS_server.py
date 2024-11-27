import cv2
from aiohttp import web
from face_recognize import recognize_face
# from Id_detection.yolov5.ID_Detection import detect_id_card

async def video_feed(request):
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={'Content-Type': 'multipart/x-mixed-replace; boundary=frame'}
    )
    await response.prepare(request)

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Apply face recognition processing
        modified_frame = recognize_face(frame)
        # modified_frame = detect_id_card(modified_frame)

        # Encode the frame as JPEG
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