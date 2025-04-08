import glob
import json
import os
import subprocess

import imageio
import numpy as np
import tyro

from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm


def convert_polycam_to_colmap_format(data_dir: str) -> None:
    output_dir = os.path.join(data_dir, "colmap")
    os.makedirs(output_dir, exist_ok=True)

    transforms_path = os.path.join(data_dir, "transforms.json")
    if not os.path.exists(transforms_path):
        raise FileNotFoundError(f"Transforms.json not found {transforms_path}")

    with open(transforms_path, "r") as file:
        data = json.load(file)
    # Extract parameters
    fx = data["frames"][0]["fl_x"]
    fy = data["frames"][0]["fl_y"]
    cx = data["frames"][0]["cx"]
    cy = data["frames"][0]["cy"]

    camera_id = 1  # Assume a single camera
    model = "PINHOLE"
    params = [fx, fy, cx, cy]

    # Get list of image files
    image_dir = os.path.join(data_dir, "images")
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_files:
        raise FileNotFoundError(f"No images found in directory: {image_dir}")

    # Get image size from the first image
    from PIL import Image

    with Image.open(image_files[0]) as img:
        width, height = img.size

    # Create output directories
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    sparse_bin_dir = os.path.join(output_dir, "sparse", "0_bin")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(sparse_bin_dir, exist_ok=True)

    # Write cameras.txt
    cameras_txt_path = os.path.join(sparse_dir, "cameras.txt")
    with open(cameras_txt_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(
            f'{camera_id} {model} {int(width)} {int(height)} {" ".join(map(str, params))}\n'
        )

    # Write images.txt
    images_txt_path = os.path.join(sparse_dir, "images.txt")
    with open(images_txt_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for idx, img_path in enumerate(image_files):
            image_id = idx + 1
            image_name = os.path.basename(img_path)

            # Read world-to-camera transformation directly from ob_in_cam (X_CW)
            # pose_idx = os.path.splitext(image_name)[0]  # Get '000000' from '000000.png'
            # pose_path = os.path.join(data_dir, "poses", f"{pose_idx}.txt")
            # if not os.path.exists(pose_path):
            #     raise FileNotFoundError(f"Pose file not found: {pose_path}")
            # X_CW = np.loadtxt(pose_path)  # Shape (4, 4)
            X_CW = data["frames"][idx]["transform_matrix"]

            # Extract rotation and translation directly
            R_CW = X_CW[:3, :3]
            p_CW = X_CW[:3, 3]

            # Convert rotation matrix to quaternion
            rot = R.from_matrix(R_CW)
            quat = rot.as_quat()  # Returns [qx, qy, qz, qw]
            qx, qy, qz, qw = quat
            # COLMAP expects [qw, qx, qy, qz]

            # Write to images.txt
            f.write(
                f"{image_id} {qw} {qx} {qy} {qz} {p_CW[0]} {p_CW[1]} {p_CW[2]} {camera_id} {image_name}\n"
            )

            # Write empty line for 2D points (since we have none)
            f.write("\n")

    # Write empty points3D.txt
    points3D_txt_path = os.path.join(sparse_dir, "points3D.txt")
    with open(points3D_txt_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("\n")

    # Convert text format to binary using COLMAP's model_converter.
    subprocess.run(
        [
            "colmap",
            "model_converter",
            "--input_path",
            sparse_dir,
            "--output_path",
            sparse_bin_dir,
            "--output_type",
            "BIN",
        ],
        check=True,
    )

    if not data_dir == output_dir:
        # Copy images to images directory
        images_output_dir = os.path.join(output_dir, "images")
        os.makedirs(images_output_dir, exist_ok=True)

        for img_path in image_files:
            shutil.copy(img_path, images_output_dir)

    logging.info("Conversion to COLMAP format completed successfully.")


if __name__ == "__main__":
    tyro.cli(convert_polycam_to_colmap_format)
