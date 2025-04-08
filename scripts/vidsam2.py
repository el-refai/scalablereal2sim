import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from segment_moving_obj_data import downsample_images

from scalable_real2sim.segmentation.segment_moving_object_data import convert_png_to_jpg


def setup_device():
    """Set up the computation device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Set GPU options if CUDA is available
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(
            "\nMPS support is preliminary. SAM 2 might give different outputs on MPS devices."
        )
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def show_mask(mask, ax, obj_id=None):
    """Display a mask on the given axes."""
    cmap = plt.get_cmap("tab10")
    color = np.array([*cmap(0 if obj_id is None else obj_id)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    """Display positive and negative points on the given axes."""
    if len(coords) == 0:
        return

    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    if len(pos_points) > 0:
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color="green",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )

    if len(neg_points) > 0:
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color="red",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )


def collect_points(image):
    """Display image and collect positive and negative points from user clicks."""
    positive_points = []  # Points to segment (right clicks)
    negative_points = []  # Points to NOT segment (left clicks)

    def onclick(event):
        if event.button == 3:  # Right click (positive points)
            positive_points.append((int(event.xdata), int(event.ydata)))
            print(f"Positive point added at: ({int(event.xdata)}, {int(event.ydata)})")
        elif event.button == 1:  # Left click (negative points)
            negative_points.append((int(event.xdata), int(event.ydata)))
            print(f"Negative point added at: ({int(event.xdata)}, {int(event.ydata)})")

        # Update the visualization
        ax.clear()
        ax.imshow(image)

        # Show current points
        if positive_points:
            ax.scatter(
                [p[0] for p in positive_points],
                [p[1] for p in positive_points],
                color="green",
                marker="*",
                s=200,
                edgecolor="white",
                linewidth=1.25,
            )
        if negative_points:
            ax.scatter(
                [p[0] for p in negative_points],
                [p[1] for p in negative_points],
                color="red",
                marker="*",
                s=200,
                edgecolor="white",
                linewidth=1.25,
            )

        ax.set_title(
            "Right click: points TO segment, Left click: points NOT to segment.\nClose window when done."
        )
        fig.canvas.draw()

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.set_title(
        "Right click: points TO segment, Left click: points NOT to segment.\nClose window when done."
    )

    # Connect the click event to the handler
    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.tight_layout()
    plt.show()

    # Disconnect the event handler
    fig.canvas.mpl_disconnect(cid)

    # Convert points to arrays and labels
    input_points = []
    input_labels = []

    if positive_points:
        input_points.extend(positive_points)
        input_labels.extend([1] * len(positive_points))

    if negative_points:
        input_points.extend(negative_points)
        input_labels.extend([0] * len(negative_points))

    if input_points:
        return np.array(input_points), np.array(input_labels, dtype=np.int32)
    else:
        return None, None


def update_depth_folders(depth_folder, rgb_folder):
    import concurrent.futures
    import shutil

    from tqdm import tqdm

    # Get full paths of depth images
    depth_files = sorted(
        [
            os.path.join(depth_folder, f)
            for f in os.listdir(depth_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    # Create directory for original depth images
    depth_original_dir = os.path.join(depth_folder, "..", "depth_original")
    os.makedirs(depth_original_dir, exist_ok=True)

    # Move original images to depth_original.
    for file_path in depth_files:
        # Use the basename since file_path is already the full path
        file_name = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(depth_original_dir, file_name))

    # Get base names of all .jpg files in the RGB folder
    rgb_basenames = {
        os.path.splitext(filename)[0]
        for filename in os.listdir(rgb_folder)
        if filename.lower().endswith(".jpg")
    }

    # Find all .png files in the depth_original_dir that have matching base names
    matching_files = []
    for filename in os.listdir(depth_original_dir):
        if filename.lower().endswith(".png"):
            base_name = os.path.splitext(filename)[0]
            if base_name in rgb_basenames:
                matching_files.append(filename)

    # Debug breakpoint (optional)
    breakpoint()

    # Copy selected images to the original depth folder
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    lambda img_name: shutil.copy(
                        os.path.join(depth_original_dir, img_name), depth_folder
                    ),
                    matching_files,
                ),
                total=len(matching_files),
                desc="Copying selected images",
            )
        )


def run_sam2_video_prediction(image_dir, output_dir=None):
    """Main function to run SAM2 video prediction with user-selected points."""
    device = setup_device()

    # Ensure output directory exists
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_dir), "masks")
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted list of images
    image_files = sorted(
        [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    # arbitrarily chosen max number of imgs before we run out of memory
    max_imgs = 1500
    if len(image_files) > max_imgs:
        downsample_images(rgb_dir=image_dir, num_images=max_imgs)
        # we need to also update the depth folders so that its looking at the right data
        update_depth_folders(
            depth_folder=os.path.join(os.path.join(video_dir, os.pardir, "depth")),
            rgb_folder=video_dir,
        )

    jpg_dir = os.path.join(image_dir, "jpg")
    convert_png_to_jpg(input_folder=image_dir, output_folder=jpg_dir)

    video_dir = jpg_dir

    # NEED TO UPDATE IMAGE_FILES!!!
    image_files = sorted(
        [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    # Load the first image
    image = np.array(Image.open(image_files[0]))
    if image.shape[2] > 3:  # Remove alpha channel if present
        image = image[:, :, :3]

    # Collect points from user
    input_points, input_labels = collect_points(image)

    if input_points is None or len(input_points) == 0:
        print("No points selected. Exiting.")
        return

    print(
        f"Selected {len(input_points)} points: {len(input_labels[input_labels==1])} positive, {len(input_labels[input_labels==0])} negative"
    )

    # Import SAM2 here to avoid loading the model until needed
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        print(
            "Error: SAM2 module not found. Make sure the SAM2 package is installed and in your Python path."
        )
        return

    # Load SAM2 model
    print("Loading SAM2 model...")
    checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Build the predictor
    try:
        predictor = build_sam2_video_predictor(model_cfg, checkpoint)
        print("SAM2 model loaded successfully")
    except Exception as e:
        print(f"Error loading SAM2 model: {e}")
        return

    # Initialize the state with the video path
    inference_state = predictor.init_state(video_path=video_dir)

    # Set the annotation frame index (first frame)
    ann_frame_idx = 0
    ann_obj_id = 1  # Object ID

    # No bounding box
    bbox = None

    # Add points to the model
    print("Running segmentation...")
    try:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=input_points,
            labels=input_labels,
            box=bbox,
        )

        # Create visualization of the initial frame segmentation
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        show_points(input_points, input_labels, plt.gca())
        show_mask(
            (out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0]
        )
        plt.title("Initial Segmentation")
        plt.axis("off")
        # plt.savefig(os.path.join("..", output_dir, "initial_segmentation.png"))
        plt.show()

        # Run propagation through the video and collect results
        print("Propagating segmentation through video...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Save masked images
        print(f"Saving masked images to {output_dir}...")
        for out_frame_idx, frame_segments in video_segments.items():
            if out_frame_idx < len(image_files):
                # Get original filename without extension
                orig_filename = os.path.basename(image_files[out_frame_idx])
                base_filename = os.path.splitext(orig_filename)[0]

                for out_obj_id, out_mask in frame_segments.items():
                    # Convert to binary float32 (0.0 or 1.0)
                    binary_mask = out_mask.squeeze(0).astype(np.float32)

                    # Save as float32 binary mask
                    cv2.imwrite(
                        os.path.join(output_dir, f"{base_filename}.png"), binary_mask
                    )
                    # np.save(os.path.join(output_dir, f"{base_filename}.npy"), binary_mask)

                # curr_img = cv2.imread(image_files[out_frame_idx])
                # for out_obj_id, out_mask in frame_segments.items():
                #     sam_mask = out_mask.squeeze(0).astype(bool)
                #     masked_img = curr_img.copy()
                #     masked_img[~sam_mask] = 0

                #     # Ensure frame index has leading zeros
                #     frame_idx_str = str(out_frame_idx).zfill(len(str(len(image_files))))
                #     # cv2.imwrite(os.path.join(output_dir, f"{frame_idx_str}.png"), masked_img)
                #     cv2.imwrite(os.path.join(output_dir, f"{frame_idx_str}.png"), sam_mask)

        print(f"Processing complete! Results saved to {output_dir}")
        print("To create a video from these images, run:")
        print(
            f"cd {output_dir} && ffmpeg -framerate 30 -i '%04d.png' -c:v libx264 -pix_fmt yuv420p output.mp4"
        )

    except Exception as e:
        print(f"Error during segmentation: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM2 Video Segmentation Tool")
    parser.add_argument("image_dir", type=str, help="Directory containing input images")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save masked images (default: image_dir/../masked_imgs)",
    )

    args = parser.parse_args()

    run_sam2_video_prediction(args.image_dir, args.output_dir)
