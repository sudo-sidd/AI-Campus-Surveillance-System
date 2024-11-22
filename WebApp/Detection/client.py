import requests
import cv2
import numpy as np
from io import BytesIO

# Server URL
SERVER_URL = "http://localhost:8000/get_frame"


def get_frame_from_server():
    try:
        # Make a GET request to the server's /get_frame endpoint
        response = requests.get(SERVER_URL)

        if response.status_code == 200:
            # If successful, convert the byte response to an image
            img_data = BytesIO(response.content)
            img_arr = np.frombuffer(img_data.read(), np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            return img
        else:
            print("Failed to fetch frame. Status Code:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error during the request: {e}")
        return None


def main():
    while True:
        frame = get_frame_from_server()

        if frame is not None:
            # Show the frame using OpenCV's imshow method
            cv2.imshow("Webcam Feed", frame)

            # Exit if the user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No frame received.")

        # Sleep briefly to control the frame rate (optional)
        # time.sleep(0.1)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
