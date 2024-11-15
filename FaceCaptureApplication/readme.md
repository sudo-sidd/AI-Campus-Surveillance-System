# Face Image Capture Application

This application allows an admin user to log in, input a unique 12-digit ID, and capture face images using a connected camera. The images are saved in a folder named after the unique ID. The app supports switching between multiple camera indices and provides a smooth user interface built with PyQt5.

## Features
- **Admin Login**: Secure login screen to access the app.
- **Unique ID Input**: Allows entering a unique 12-digit ID to create a folder for the user.
- **Camera Feed**: Live video feed from the camera to guide users during the image capture process.
- **Image Capture**: Captures 5 images of the user’s face from different angles.
- **Camera Selection**: Ability to change the camera index from the settings menu.
- **Error and Success Messages**: Informative pop-ups to guide the user throughout the process.

## Requirements
- Python 3.x
- PyQt5
- OpenCV (cv2)

You can install the necessary dependencies using `pip`:

```bash
pip install pyqt5 opencv-python
```

## How to Run

1. Clone or download this repository to your local machine.
2. Install the required dependencies as mentioned above.
3. Run the script:

```bash
python capture_app.py
```

## Functionality Walkthrough

### 1. Login Screen
When you launch the app, you will see the **Login Screen**. To proceed, enter the following credentials:
- **Username**: admin
- **Password**: 1234

Once logged in, the app will navigate to the main UI.

### 2. Main UI
In the **Main UI**, you will be prompted to enter a **Unique ID**. The ID must be exactly 12 digits long. Once entered, the app will check if a folder already exists for that ID. If not, it will create the folder and prepare to capture images.

### 3. Image Capture Process
- The app will display a live camera feed and prompt you to position your face in front of the camera.
- It will capture 5 images (one for each side of your face).
- After each capture, the app will ask you to position your face for the next image.
- Once all 5 images are captured, the app will display a completion message and stop the camera feed.

### 4. Camera Selection (Settings Menu)
- You can change the camera index by selecting the **Settings** menu from the top menu bar.
- A dialog will allow you to input a new camera index (between 0 and 10).

### 5. Error Handling
If the camera is not accessible or any other error occurs, the app will show an error message to guide the user.

## Folder Structure
When a new user is created (with a valid 12-digit ID), the app will create a folder inside the `./users/` directory with the following structure:

```
./users/{unique_id}/
    face_1.jpg
    face_2.jpg
    face_3.jpg
    face_4.jpg
    face_5.jpg
```

Each image will be saved with a name like `face_1.jpg`, `face_2.jpg`, etc., representing different angles of the user's face.

## Known Issues
- The app assumes that the camera is properly connected and accessible. If not, an error message will be displayed.
- The app does not currently handle other exceptions that may arise from the OpenCV library.

