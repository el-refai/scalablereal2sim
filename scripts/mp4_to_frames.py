import argparse
import os

import cv2


def extract_frames(video_path, output_folder):
    """
    Extract frames from a video file and save them as PNG images.

    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to the output folder
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    print(f"Video info: {frame_count} frames, {fps} FPS")

    # Process video frame by frame
    count = 0
    success = True

    while success:
        # Read the next frame
        success, frame = video.read()

        if success:
            # Save the frame as a PNG file with 6-digit numbering
            output_path = os.path.join(output_folder, f"{count:06d}.png")
            cv2.imwrite(output_path, frame)

            # Increment counter
            count += 1

            # Print progress every 100 frames
            if count % 100 == 0:
                print(f"Processed {count} frames out of {frame_count}")

    video.release()
    print(f"Extraction complete. {count} frames extracted to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument(
        "--output_folder", default="frames", help="Path to the output folder"
    )

    args = parser.parse_args()
    extract_frames(args.video_path, args.output_folder)
