import os
import cv2
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtMultimedia import QSound  # Import QSound for sound playback



class CaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initial variables
        self.unique_id = None
        self.image_count = 0
        self.capture = None
        self.camera_index = 0  # Default camera index
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Load face detector
        self.capture_images = []
        self.current_frame = None
        self.sound_file = "Face_rec-ID_detection/FaceCaptureApplication/camera-shutter.wav"

        # Instructions for each image capture
        self.instructions = [
            "Look directly at the camera",
            "Turn your head 45° to the left",
            "Turn your head 90° to the left",
            "Turn your head 45° to the right",
            "Turn your head 90° to the right",
            "Slightly lower your head",
            "Lower your head and turn 45° to the right",
            "Lower your head and turn 45° to the left",
            "Slightly raise your head"
        ]

        # Basic Setup
        self.setWindowTitle("Face Image Capture")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #ffffff; font-family: Arial; font-size: 12pt;")

        # Main Layout
        self.layout = QVBoxLayout()
            # Initialize Login UI
   

        self.init_ui()

        # Central Widget
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)
        

 
    def init_ui(self):
        """Set up the UI"""
        self.label = QLabel("Enter Unique ID (12 digits):")
        self.label.setFont(QFont("Arial", 14))
        self.layout.addWidget(self.label)

        self.text_input = QLineEdit()
        self.text_input.setMaxLength(12)
        self.text_input.setPlaceholderText("Unique ID (12 digits)")
        self.text_input.setStyleSheet("padding: 10px; font-size: 14pt; border: 1px solid #ccc; border-radius: 5px;")
        self.layout.addWidget(self.text_input)

        self.submit_button = QPushButton("Submit")
        self.submit_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 12px; border-radius: 5px; font-size: 14pt;")
        self.submit_button.clicked.connect(self.on_submit)
        self.layout.addWidget(self.submit_button)

        self.graphics_view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)
        self.graphics_view.setFixedSize(900, 500)
        self.graphics_view.setStyleSheet("background-color: #000; border: 2px solid #ccc; border-radius: 10px;")
        self.layout.addWidget(self.graphics_view)

        self.face_position_label = QLabel("Face Position Feedback: ")
        self.face_position_label.setFont(QFont("Arial", 12))
        self.face_position_label.setStyleSheet("padding: 5px; border: 1px solid #ccc;")
        self.layout.addWidget(self.face_position_label)

        # Revert Button
        self.revert_button = QPushButton("Revert Last Capture")
        self.revert_button.setStyleSheet("background-color: #f44336; color: white; padding: 12px; border-radius: 5px; font-size: 14pt;")
        self.revert_button.clicked.connect(self.revert_capture)
        self.layout.addWidget(self.revert_button)

    def keyPressEvent(self, event):
        """Handle key press events for capturing, reverting, and other actions."""
        if event.key() == Qt.Key_Shift and self.capture is not None and self.current_frame is not None:
            # Right arrow key captures an image
            self.revert_capture()
        elif event.key() == Qt.Key_Apostrophe:
            # Left arrow key reverts the last capture
            self.revert_capture()
        elif event.key() == Qt.Key_Return:  # Main Enter key
            self.capture_image()
        elif event.key() == Qt.Key_Enter:  # Numeric keypad Enter key
           self.capture_image()



    def on_submit(self):
        """Handle Unique ID submission"""
        unique_id = self.text_input.text()
        if len(unique_id) != 12 or not unique_id.isdigit():
            self.show_error("Invalid ID", "ID must be exactly 12 digits.")
            return

        self.unique_id = unique_id
        self.create_user_folder()

    def create_user_folder(self):
        """Create folder for the user with unique ID"""
        folder_path = f'./users/{self.unique_id}'
        if os.path.exists(folder_path):
            self.show_error("Folder Exists", f"Folder for ID {self.unique_id} already exists.")
            return

        os.makedirs(folder_path)
        self.show_message("Folder Created", f"Folder for ID {self.unique_id} created. Ready to capture images.")
        self.start_camera()

    def start_camera(self):
        """Initialize camera and start capturing images"""
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            self.show_error("Camera Error", "Unable to access the camera.")
            return

        self.capture_images = [f"face_{i}.jpg" for i in range(1, 10)]  # Image names for 9 captures
        self.image_count = 0
        self.update_feedback()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        """Update the live camera feed and detect faces"""
        if self.capture is not None:
            ret, frame = self.capture.read()
            if not ret:
                return

            # Convert frame to QPixmap
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = frame_rgb.shape
            image = QImage(frame_rgb.data, width, height, width * 3, QImage.Format_RGB888)
            pixmap = QPixmap(image)

            if not hasattr(self, 'video_item') or self.video_item is None:
                self.video_item = QGraphicsPixmapItem(pixmap)
                self.scene.addItem(self.video_item)
            else:
                self.video_item.setPixmap(pixmap)

            self.current_frame = frame  # Save current frame for capturing

    def update_feedback(self):
        """Update face position feedback"""
        if self.image_count < len(self.instructions):
            self.face_position_label.setText(f"Instruction: {self.instructions[self.image_count]}")
        else:
            self.face_position_label.setText("All images captured.")

    def capture_image(self):
        """Capture and save an image when the Enter key is pressed"""
        if self.capture is not None and hasattr(self, 'current_frame'):
            frame = self.current_frame
            image_path = f'./users/{self.unique_id}/{self.capture_images[self.image_count]}'
            cv2.imwrite(image_path, frame)
            
            # Play sound
            QSound.play(self.sound_file)

            self.image_count += 1
            self.update_feedback()

            if self.image_count == len(self.instructions):
                self.show_message("Capture Complete", "You have successfully captured all images.")
                self.reset_ui()

    def revert_capture(self):
        """Revert the last captured image"""
        if self.image_count > 0:
            self.image_count -= 1
            image_path = f'./users/{self.unique_id}/{self.capture_images[self.image_count]}'
            if os.path.exists(image_path):
                os.remove(image_path)
            self.update_feedback()
            # self.show_message("Revert Successful", "Last capture has been reverted. You can capture again.")

    def reset_ui(self):
        """Reset the UI and release the camera"""
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.unique_id = None
        self.text_input.clear()
        self.scene.clear()
        self.face_position_label.setText("Face Position Feedback: ")
        self.video_item = None

    def show_error(self, title, message):
        """Show error message"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStyleSheet("QMessageBox {font-family: Arial; font-size: 12pt;}")
        msg.exec_()

    def show_message(self, title, message):
        """Show success or informational message"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStyleSheet("QMessageBox {font-family: Arial; font-size: 12pt;}")
        msg.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CaptureApp()
    window.show()
    sys.exit(app.exec_())
