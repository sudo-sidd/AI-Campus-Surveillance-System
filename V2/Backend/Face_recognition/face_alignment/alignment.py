import cv2
import numpy as np
import mediapipe as mp
from skimage import transform as trans

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    refine_landmarks=True
)

# Define standard destination points for face alignment (based on a 128x128 image)
# These points represent: [left eye center, right eye center, nose tip, left mouth corner, right mouth corner]
DEST_LANDMARKS = np.array([
    [38.2946, 51.6963],   # Left eye center
    [73.5318, 51.5014],   # Right eye center
    [56.0252, 71.7366],   # Nose tip
    [41.5493, 92.3655],   # Left mouth corner
    [70.7299, 92.2041],   # Right mouth corner
], dtype=np.float32)

# MediaPipe landmark indices mapping to our 5 key points
# Based on the MediaPipe face mesh topology
# https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
LEFT_EYE_CENTER = 159       # Left eye center point 
RIGHT_EYE_CENTER = 386      # Right eye center point
NOSE_TIP = 1                # Nose tip
LEFT_MOUTH_CORNER = 61      # Left mouth corner
RIGHT_MOUTH_CORNER = 291    # Right mouth corner

# Landmark indices for eye, nose, and mouth
LANDMARK_INDICES = [LEFT_EYE_CENTER, RIGHT_EYE_CENTER, NOSE_TIP, 
                   LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER]

# Define a standard set of destination landmarks for ArcFace alignment
arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_norm(lmk, image_size=112, mode="arcface"):
    """
    Estimate the transformation matrix for aligning facial landmarks.

    Args:
        lmk (numpy.ndarray): 2D array of shape (5, 2) representing facial landmarks.
        image_size (int): Desired output image size.
        mode (str): Alignment mode, currently only "arcface" is supported.

    Returns:
        numpy.ndarray: Transformation matrix (2x3) for aligning facial landmarks.
    """
    # Check input conditions
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0

    # Adjust ratio and x-coordinate difference based on image size
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    # Scale and shift the destination landmarks
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x

    # Estimate the similarity transformation
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]

    return M


def norm_crop(img, landmark, image_size=112, mode="arcface"):
    """
    Normalize and crop a facial image based on provided landmarks.

    Args:
        img (numpy.ndarray): Input facial image.
        landmark (numpy.ndarray): 2D array of shape (5, 2) representing facial landmarks.
        image_size (int): Desired output image size.
        mode (str): Alignment mode, currently only "arcface" is supported.

    Returns:
        numpy.ndarray: Normalized and cropped facial image.
    """
    # Estimate the transformation matrix
    M = estimate_norm(landmark, image_size, mode)

    # Apply the affine transformation to the image
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

    return warped

def get_landmarks(image):
    """
    Extract 5-point facial landmarks using MediaPipe.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        landmarks: Array of shape (5, 2) containing landmark coordinates or None if no face detected
    """
    try:
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Process the image with MediaPipe
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
            
        # Extract the specified landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        points = []
        
        for idx in LANDMARK_INDICES:
            x = landmarks[idx].x * width
            y = landmarks[idx].y * height
            points.append([x, y])
        
        return np.array(points, dtype=np.float32)
        
    except Exception as e:
        print(f"Error extracting landmarks: {e}")
        return None

def align_face(image, landmarks=None, output_size=128):
    """
    Align face using landmarks or detect them if not provided.
    
    Args:
        image: Input facial image (BGR format)
        landmarks: Optional pre-detected landmarks
        output_size: Size of output aligned face (default: 128)
        
    Returns:
        Aligned face image of specified size
    """
    if landmarks is None:
        landmarks = get_landmarks(image)
    
    if landmarks is None:
        # Fallback to center crop if landmark detection fails
        return center_crop_face(image, output_size)
    
    # Scale destination landmarks to match output size
    scale = output_size / 128.0
    dest_landmarks = DEST_LANDMARKS * scale
    
    # Calculate similarity transform
    M = cv2.estimateAffinePartial2D(landmarks, dest_landmarks)[0]
    
    # Apply transformation
    aligned_face = cv2.warpAffine(image, M, (output_size, output_size), 
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return aligned_face

def center_crop_face(image, output_size=128):
    """
    Simple center crop and resize as fallback alignment method.
    
    Args:
        image: Input facial image
        output_size: Size of output aligned face
        
    Returns:
        Center-cropped and resized face
    """
    h, w = image.shape[:2]
    size = min(h, w)
    
    # Calculate crop coordinates
    x_center = w // 2
    y_center = h // 2
    x1 = max(0, x_center - size // 2)
    y1 = max(0, y_center - size // 2)
    x2 = min(w, x_center + size // 2)
    y2 = min(h, y_center + size // 2)
    
    # Crop and resize
    cropped = image[y1:y2, x1:x2]
    aligned = cv2.resize(cropped, (output_size, output_size))
    
    return aligned

def enhance_image(image):
    """
    Enhance the quality of a facial image.
    
    Args:
        image: Input facial image
        
    Returns:
        Enhanced image with better contrast
    """
    try :
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        
        # Merge back the enhanced luminance with original color
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply subtle noise reduction
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
        
        return enhanced
    
    except Exception as e :
        print(f"Error enhancing image: {e}")
        return image
    
