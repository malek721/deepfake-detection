import cv2
import os
import numpy as np


def crop_center(img, target_size=256):
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped_img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    resized_img = cv2.resize(cropped_img, (target_size, target_size))
    return resized_img


def video_to_frames(video_path, output_folder, frame_rate=5, target_size=256):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        if frame_count % frame_rate == 0:
            processed_frame = crop_center(frame, target_size)
            frame_filename = os.path.join(video_output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, processed_frame)
            saved_frame_count += 1

        frame_count += 1


        if frame_count % 1000 == 0:
            print(f"Processed {frame_count}/{total_frames} frames of video {video_name}")

    cap.release()
    print(f"Frames extracted and saved to {video_output_folder}")


def process_multiple_videos(videos_folder, output_folder, frame_rate=5, target_size=256):

    video_files = [f for f in os.listdir(videos_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    for video_file in video_files:
        video_path = os.path.join(videos_folder, video_file)
        video_to_frames(video_path, output_folder, frame_rate, target_size)
        print(f"Finished processing {video_file}")



videos_folder = r"C:\Users\admin\Desktop\yapa zeka\data\original"
output_folder = r"C:\Users\admin\Desktop\yapa zeka\croped_original"
frame_rate = 1
target_size = 256

process_multiple_videos(videos_folder, output_folder, frame_rate, target_size)
