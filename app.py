from flask import Flask, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)  # static_image_mode for consistency

def draw_keypoints(image, keypoints, connections):
    for connection in connections:
        start = keypoints[connection[0]]
        end = keypoints[connection[1]]
        start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
        end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        cv2.circle(image, start_point, 5, (0, 0, 255), -1)
        cv2.circle(image, end_point, 5, (0, 0, 255), -1)
    return image

def draw_angles(image, keypoints, angles):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for angle in angles:
        p1, p2, p3, angle_value = angle
        p1_coord = (int(keypoints[p1].x * image.shape[1]), int(keypoints[p1].y * image.shape[0]))
        p2_coord = (int(keypoints[p2].x * image.shape[1]), int(keypoints[p2].y * image.shape[0]))
        p3_coord = (int(keypoints[p3].x * image.shape[1]), int(keypoints[p3].y * image.shape[0]))

        # Draw lines between keypoints
        cv2.line(image, p1_coord, p2_coord, (0, 255, 0), 2)
        cv2.line(image, p2_coord, p3_coord, (0, 255, 0), 2)

        # Draw angle value
        cv2.putText(image, str(int(angle_value)), p2_coord, font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    return image

def calculate_angle(p1, p2, p3):
    def to_array(point):
        if isinstance(point, tuple):
            return np.array(point)
        return np.array([point.x, point.y])
    
    a = to_array(p1)
    b = to_array(p2)
    c = to_array(p3)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def is_plank_form_correct(keypoints):
    left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_ankle = keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    # Calculate midpoint of shoulders and hips
    mid_shoulder = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
    mid_hip = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
    mid_ankle = ((left_ankle.x + right_ankle.x) / 2, (left_ankle.y + right_ankle.y) / 2)

    # Calculate spine angle as the angle between mid_shoulder, mid_hip, and mid_ankle
    spine_angle = calculate_angle(mid_shoulder, mid_hip, mid_ankle)

    left_forearm_angle = calculate_angle(left_shoulder, keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value], keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value])
    right_forearm_angle = calculate_angle(right_shoulder, keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value], keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    print(f"Left forearm angle: {left_forearm_angle}")
    print(f"Right forearm angle: {right_forearm_angle}")
    print(f"Spine angle: {spine_angle}")

    # Adjust thresholds based on the visualization results
    if (
        (80 < left_forearm_angle < 115 or
        80 < right_forearm_angle < 115) and
        160 < spine_angle < 180
    ):
        return True
    return False

def analyze_image(file_path):
    try:
        print(f"Analyzing image: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        image = cv2.imread(file_path)
        if image is None:
            raise Exception("Failed to read image")

        print("Image read successfully.")
        print(f"Image shape: {image.shape}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            keypoints = results.pose_landmarks.landmark
            # Calculate angles
            left_forearm_angle = calculate_angle(
                keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value]
            )
            right_forearm_angle = calculate_angle(
                keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            )
            body_angle = calculate_angle(
                keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                keypoints[mp_pose.PoseLandmark.LEFT_HIP.value],
                keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            )

            angles = [
                (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value, left_forearm_angle),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value, right_forearm_angle),
                (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_ANKLE.value, body_angle)
            ]

            # Draw keypoints, connections, and angles for visualization
            connections = [
                (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
                (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
                (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value),
                (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
                (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
            ]
            image_with_keypoints = draw_keypoints(image, keypoints, connections)
            image_with_angles = draw_angles(image_with_keypoints, keypoints, angles)
            cv2.imwrite('/tmp/keypoints_visualization_with_angles.jpg', image_with_angles)
            print("Keypoints and angles visualization saved to /tmp/keypoints_visualization_with_angles.jpg")

            return is_plank_form_correct(keypoints)
        else:
            print("Pose landmarks not detected.")
            raise Exception("Pose landmarks not detected")

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        return False

    except Exception as e:
        print(f"Exception: {e}")
        return False

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'}), 400

        upload_folder = os.path.join(os.getcwd(), 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Verify that the file was saved correctly
        if not os.path.exists(file_path):
            print(f"Failed to save file at: {file_path}")
            return jsonify({'success': False, 'message': 'Failed to save file'}), 500

        result = analyze_image(file_path)

        os.remove(file_path)  # Clean up the saved file

        if result:
            return jsonify({'success': True, 'isCorrect': True})
        else:
            return jsonify({'success': True, 'isCorrect': False, 'message': 'Your form is incorrect. Please review proper plank form techniques.'})
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'success': False, 'message': 'Error analyzing the image'}), 500

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
