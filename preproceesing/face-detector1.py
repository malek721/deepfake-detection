import cv2
import os
import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_and_crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return frame

    (x, y, w, h) = faces[0]
    cropped_face = frame[y:y + h, x:x + w]
    return cropped_face


def resize_with_padding(image, target_size):
    h, w = image.shape[:2]
    if h > w:
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        new_w = target_size
        new_h = int(h * target_size / w)
    resized_img = cv2.resize(image, (new_w, new_h))
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    final_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return final_img


def video_to_frames_and_process(video_path, output_folder, frame_rate=1, target_size=256):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)
    frame_count = 0
    saved_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:

            processed_frame = detect_and_crop_face(frame)
            resized_frame = resize_with_padding(processed_frame, target_size)
            frame_filename = os.path.join(video_output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, resized_frame)
            saved_frame_count += 1
        frame_count += 1
        if frame_count % 1000 == 0:
            print(f"Processed {frame_count}/{total_frames} frames of video {video_name}")
    cap.release()
    print(f"Frames extracted, processed, and saved to {video_output_folder}")


def process_multiple_videos(videos_folder, output_folder, frame_rate=1, target_size=256):
    video_files = [f for f in os.listdir(videos_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    for video_file in video_files:
        video_path = os.path.join(videos_folder, video_file)
        video_to_frames_and_process(video_path, output_folder, frame_rate, target_size)
        print(f"Finished processing {video_file}")


videos_folder = r"C:\Users\admin\Desktop\yapa zeka\data\fake"
output_folder = r"C:\Users\admin\Desktop\yapa zeka\processed_fake"
frame_rate = 5
target_size = 256

process_multiple_videos(videos_folder, output_folder, frame_rate, target_size)