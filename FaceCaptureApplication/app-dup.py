import os
import cv2
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMenuBar, QMenu, QAction, QInputDialog
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import QTimer, Qt


class CaptureApp(QMainWindow):  # Subclass QMainWindow instead of QWidget
    def __init__(self):
        super().__init__()

        # Initial variables
        self.unique_id = None
        self.image_count = 0
        self.capture = None
        self.capture_images = []
        self.camera_index = 0  # Default camera index

        # Basic Setup
        self.setWindowTitle("Face Image Capture")
        self.setGeometry(100, 100, 800, 600)  # Set window size
        self.setStyleSheet("background-color: #ffffff; font-family: Arial; font-size: 12pt;")

        # Main Layout
        self.layout = QVBoxLayout()

        # Initialize Login UI
        self.init_login_ui()

        # Initialize Main UI (hidden initially)
        self.init_main_ui()

        # Central Widget
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

    def init_login_ui(self):
        """Set up the login screen UI"""
        self.login_widget = QWidget()
        self.login_layout = QVBoxLayout()

        # Title Label
        self.login_label = QLabel("Admin Login")
        self.login_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.login_label.setAlignment(Qt.AlignCenter)
        self.login_label.setStyleSheet("color: #333; margin-bottom: 20px;")
        self.login_layout.addWidget(self.login_label)

        # Username Input
        self.label_username = QLabel("Username:")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter username")
        self.username_input.setStyleSheet("padding: 10px; margin-bottom: 10px; font-size: 14pt; border: 1px solid #ccc; border-radius: 5px;")

        # Password Input
        self.label_password = QLabel("Password:")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Enter password")
        self.password_input.setStyleSheet("padding: 10px; margin-bottom: 20px; font-size: 14pt; border: 1px solid #ccc; border-radius: 5px;")

        # Login Button
        self.submit_button = QPushButton("Login")
        self.submit_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 12px; border-radius: 5px; font-size: 14pt;")
        self.submit_button.clicked.connect(self.on_login)

        # Add widgets to layout
        self.login_layout.addWidget(self.label_username)
        self.login_layout.addWidget(self.username_input)
        self.login_layout.addWidget(self.label_password)
        self.login_layout.addWidget(self.password_input)
        self.login_layout.addWidget(self.submit_button)

        self.login_widget.setLayout(self.login_layout)
        self.layout.addWidget(self.login_widget)

    def init_main_ui(self):
        """Set up the main UI screen (hidden initially)"""
        self.main_ui_widget = QWidget()
        self.main_ui_layout = QVBoxLayout()

        # Main Screen Label
        self.main_label = QLabel("Enter Unique ID (12 digits):")
        self.main_label.setFont(QFont("Arial", 14))
        self.main_ui_layout.addWidget(self.main_label)

        # Unique ID Input
        self.text_input = QLineEdit()
        self.text_input.setMaxLength(12)
        self.text_input.setPlaceholderText("Unique ID (12 digits)")
        self.text_input.setStyleSheet("padding: 10px; font-size: 14pt; border: 1px solid #ccc; border-radius: 5px;")
        self.main_ui_layout.addWidget(self.text_input)

        # Submit Button for ID
        self.submit_button_main = QPushButton("Submit")
        self.submit_button_main.setStyleSheet("background-color: #4CAF50; color: white; padding: 12px; border-radius: 5px; font-size: 14pt;")
        self.submit_button_main.clicked.connect(self.on_submit)
        self.main_ui_layout.addWidget(self.submit_button_main)

        # Graphics View for Video
        self.graphics_view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)
        self.graphics_view.setFixedSize(640, 480)
        self.graphics_view.setStyleSheet("background-color: #000; border: 2px solid #ccc; border-radius: 10px;")
        self.main_ui_layout.addWidget(self.graphics_view)

        # Settings Menu
        self.menu_bar = self.menuBar()  # Use self.menuBar() instead of setting it manually
        self.settings_menu = QMenu("Settings", self.menu_bar)
        self.change_camera_action = QAction(QIcon("icons/camera.png"), "Change Camera Index", self)
        self.change_camera_action.triggered.connect(self.show_change_camera_dialog)
        self.settings_menu.addAction(self.change_camera_action)
        self.menu_bar.addMenu(self.settings_menu)

        # Set the main layout to the UI
        self.main_ui_widget.setLayout(self.main_ui_layout)

    def show_main_ui(self):
        """Hide login UI and show main UI"""
        self.login_widget.setVisible(False)
        self.main_ui_widget.setVisible(True)

        # Set the central widget
        self.setCentralWidget(self.main_ui_widget)

        # Set up camera
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            self.show_error("Camera Error", "Unable to access the camera.")
            return

        # Start the video update loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms

    def on_login(self):
        """Handle login logic"""
        username = self.username_input.text()
        password = self.password_input.text()

        if username == "admin" and password == "1234":
            self.show_message("Login Successful", "Welcome to the Capture App!")
            self.show_main_ui()
        else:
            self.show_error("Login Failed", "Invalid Username or Password")

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

        # Initialize image names
        self.capture_images = [f"face_{i}.jpg" for i in range(1, 6)]  # Image names for 5 sides
        self.capture_images_for_face()

    def capture_images_for_face(self):
        """Prompt user to position their face in front of the camera"""
        self.show_message("Position your face", "Please position your face in front of the camera.")
        self.capture_image()

    def capture_image(self):
        """Capture and save an image"""
        if self.capture is not None:
            ret, frame = self.capture.read()
            if ret:
                image_path = f'./users/{self.unique_id}/{self.capture_images[self.image_count]}'
                cv2.imwrite(image_path, frame)
                self.image_count += 1
                if self.image_count < 5:
                    self.show_message("Next Position", f"Please position your face for side {self.image_count + 1}.")
                    self.capture_image()
                else:
                    self.show_message("Capture Complete", "You have successfully captured all required images.")
                    self.capture.release()
                    self.capture = None
                    self.timer.stop()

    def update_frame(self):
        """Update the live camera feed"""
        if self.capture is not None:
            ret, frame = self.capture.read()
            if not ret:
                return

            # Convert frame to QPixmap
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = frame_rgb.shape
            image = QImage(frame_rgb.data, width, height, width * 3, QImage.Format_RGB888)
            pixmap = QPixmap(image)

            if not hasattr(self, 'video_item'):
                self.video_item = QGraphicsPixmapItem(pixmap)
                self.scene.addItem(self.video_item)
            else:
                self.video_item.setPixmap(pixmap)

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

    def show_change_camera_dialog(self):
        """Change the camera index"""
        camera_index, ok = QInputDialog.getInt(self, "Change Camera Index", "Enter Camera Index:", self.camera_index, 0, 10, 1)
        if ok:
            self.camera_index = camera_index
            self.show_message("Camera Index Changed", f"Camera index is now set to {self.camera_index}.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CaptureApp()
    window.show()
    sys.exit(app.exec_())
