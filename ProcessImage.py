import cv2
import mediapipe as mp
import os
from pathlib import Path

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Function to extract and save MediaPipe landmarks
def extract_mediapipe_landmarks(source_folder, destination_folder):
    # Create destination folder if it doesn't exist
    destination_folder.mkdir(parents=True, exist_ok=True)

    # Iterate over each emotion folder (0-7)
    for emotion_folder in source_folder.iterdir():
        if emotion_folder.is_dir():
            # Create corresponding folder in the destination
            dest_emotion_folder = destination_folder / emotion_folder.name
            dest_emotion_folder.mkdir(parents=True, exist_ok=True)

            # Iterate over images in the emotion folder
            for image_file in emotion_folder.glob("*.*"):
                # Read the RGB image
                image = cv2.imread(str(image_file))
                if image is None:
                    print(f"Unable to read image: {image_file}")
                    continue

                # Convert the image to RGB for MediaPipe
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Perform facial landmark detection using MediaPipe
                with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
                    results = face_mesh.process(rgb_image)
                    if not results.multi_face_landmarks:
                        print(f"No landmarks found for image: {image_file}")
                        continue

                    # Draw facial landmarks on the image
                    annotated_image = image.copy()
                    for face_landmarks in results.multi_face_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=1))

                    # Save the annotated image to the destination folder
                    output_path = dest_emotion_folder / f"{image_file.stem}_landmarks.jpg"
                    cv2.imwrite(str(output_path), annotated_image)

    print("Landmark extraction completed and images saved in destination folder.")


# Define source and destination folders
source_folder = Path("./affectnet_3750subset/train")
destination_folder = Path("./affectnet_landmark")

# Run the landmark extraction function
extract_mediapipe_landmarks(source_folder, destination_folder)
