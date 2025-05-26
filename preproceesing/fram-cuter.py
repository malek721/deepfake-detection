import cv2
import os


def video_to_frames(video_path, output_folder, frame_rate=1):
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
            frame_filename = os.path.join(video_output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

        if frame_count % 1000 == 0:
            print(f"Processed {frame_count}/{total_frames} frames of video {video_name}")

    cap.release()
    print(f"Frames extracted and saved to {video_output_folder}")


def process_multiple_videos(videos_folder, output_folder, frame_rate=1):
    video_files = [f for f in os.listdir(videos_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_file in video_files:
        video_path = os.path.join(videos_folder, video_file)
        video_to_frames(video_path, output_folder, frame_rate)
        print(f"Finished processing {video_file}")

videos_folder = r"C:\Users\admin\Desktop\yapa zeka\data\Deepfakes"
output_folder = r"C:\Users\admin\Desktop\yapa zeka\fakdeata"
frame_rate = 5
process_multiple_videos(videos_folder, output_folder, frame_rate)
