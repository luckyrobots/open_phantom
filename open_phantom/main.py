import os

from hand_processor import HandProcessor


def main():
    processor = HandProcessor()

    record_option = input("Record a new video? (y/n): ")

    if record_option.lower() == "y":
        print("Starting video recording...")
        video_path = processor.record_video()
        if not video_path:
            print("Video recording failed.")
            return

        print(f"Video recorded successfully to {video_path}")
        process_option = input("Process the video now? (y/n): ")
        if process_option.lower() == "n":
            print("Video processing skipped.")
            return
    else:
        video_path = input("Enter the path to a video file: ")
        while not os.path.exists(video_path):
            print("Error: Video file not found.")
            video_path = input("Enter the path to the video file: ")

    processor.process_video(video_path)


if __name__ == "__main__":
    main()
