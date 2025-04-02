import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tyro

np.random.seed(3)

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

# Set the GPU number (e.g., GPU 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

YUMI_LINKS = [
    "world",
    "base_link",
    "yumi_link_1_r",
    "yumi_link_2_r",
    "yumi_link_7_r",
    "yumi_link_3_r",
    "yumi_link_4_r",
    "yumi_link_5_r",
    "yumi_link_6_r",
    "yumi_link_1_l",
    "yumi_link_2_l",
    "yumi_link_7_l",
    "yumi_link_3_l",
    "yumi_link_4_l",
    "yumi_link_5_l",
    "yumi_link_6_l",
    "gripper_r_base",
    "gripper_r_finger_r",
    "gripper_r_finger_l",
    "gripper_l_base",
    "gripper_l_finger_l",
    "gripper_l_finger_r",
]

YUMI_RIGHT_NO_POV_POSE = {
    "yumi_joint_1_r": -2.941,
    "yumi_joint_2_r": 0.39,
    "yumi_joint_7_r": 0.0,
    "yumi_joint_3_r": 0.0,
    "yumi_joint_4_r": 0.0,
    "yumi_joint_5_r": 0.0,
    "yumi_joint_6_r": 0.0,
    "yumi_joint_1_l": -1.24839656,
    "yumi_joint_2_l": -1.09802876,
    "yumi_joint_7_l": 1.06634394,
    "yumi_joint_3_l": 0.31386161 - 0.2,
    "yumi_joint_4_l": 1.90125141,
    "yumi_joint_5_l": 1.3205139,
    "yumi_joint_6_l": 2.43563939,
    "gripper_r_joint": 0.0,  # 0.025,
    "gripper_l_joint": 0.0,  # 0.004, # 0.025,
}

YUMI_REST_POSE = {
    "yumi_joint_1_r": 1.21442839,
    "yumi_joint_2_r": -1.03205606,
    "yumi_joint_7_r": -1.10072738,
    "yumi_joint_3_r": 0.2987352 - 0.2,
    "yumi_joint_4_r": -1.85257716,
    "yumi_joint_5_r": 1.25363652,
    "yumi_joint_6_r": -2.42181893,
    "yumi_joint_1_l": -1.24839656,
    "yumi_joint_2_l": -1.09802876,
    "yumi_joint_7_l": 1.06634394,
    "yumi_joint_3_l": 0.31386161 - 0.2,
    "yumi_joint_4_l": 1.90125141,
    "yumi_joint_5_l": 1.3205139,
    "yumi_joint_6_l": 2.43563939,
    "gripper_r_joint": 0,  # 0.025,
    "gripper_l_joint": 0,  # 0.025,
}

YUMI_LEFT_JOINTS = [
    "yumi_joint_1_l",
    "yumi_joint_2_l",
    "yumi_joint_7_l",
    "yumi_joint_3_l",
    "yumi_joint_4_l",
    "yumi_joint_5_l",
    "yumi_joint_6_l",
]


def show_mask_vid(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()
        # if i == 0:
        #     plt.savefig("/home/inhand/inhand/out_data/test.png")


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


# Function to save the mask
def save_mask(mask, save_path):
    # Remove the channel dimension if it's there
    # mask = mask.squeeze(0)  # shape becomes (256, 256)

    # Convert boolean mask to uint8 (0 or 255)
    mask_img = (mask * 255).astype(np.uint8)

    # Create an Image object and save it
    mask_image = Image.fromarray(mask_img)
    mask_image.save(save_path)


# Function to merge masks and save the combined mask
def merge_and_save_masks(masks, save_path):
    # Assuming all masks have the same shape, get the shape of one mask
    combined_mask = np.zeros_like(list(masks.values())[0], dtype=np.uint8)

    # Merge all masks together
    for mask in masks.values():
        combined_mask = np.maximum(combined_mask, mask.astype(np.uint8))

    # Convert to an image and save
    combined_mask_img = Image.fromarray(
        combined_mask[0] * 255
    )  # Scale boolean to 0-255
    combined_mask_img.save(save_path)


def extract_clip_id(file_name):
    return int(file_name.split("_")[-1])


def get_bbox(mask):
    x_coords, y_coords = np.where(mask)
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    return [y_min, x_min, y_max, x_max]


def get_obbox(mask):
    """Get the oriented bounding box of the mask"""
    # Get the coordinates of all non-zero pixels in the mask
    non_zero_coords = np.argwhere(mask)
    # Get the minimum rotated rectangle that bounds the non-zero pixels
    rect = cv2.minAreaRect(non_zero_coords)
    # Get the four corners of the rectangle
    box = cv2.boxPoints(rect)
    # Convert the box to integer values
    # box = np.int(box)
    return box


def sample_points_centered_major_axis(corners, num_points=10):
    """
    Sample points uniformly along the major axis of an oriented bounding box,
    shifted to be centered across the minor axis.

    Parameters:
    - corners: Array of 4 points defining the oriented bounding box (Nx2, where N=4).
    - num_points: Number of points to sample along the major axis.

    Returns:
    - sampled_points: List of sampled points [(x1, y1), (x2, y2), ...].
    """
    # Calculate the center of the bounding box
    center = np.mean(corners, axis=0)

    # Calculate the distances and find the major and minor axes
    edges = [
        (np.linalg.norm(corners[0] - corners[1]), corners[0], corners[1]),
        (np.linalg.norm(corners[1] - corners[2]), corners[1], corners[2]),
        (np.linalg.norm(corners[2] - corners[3]), corners[2], corners[3]),
        (np.linalg.norm(corners[3] - corners[0]), corners[3], corners[0]),
    ]
    sorted_edges = sorted(edges, key=lambda x: x[0], reverse=True)
    major_axis = sorted_edges[0][1:3]  # Longest edge
    minor_axis = sorted_edges[1][1:3]  # Second longest edge

    # Calculate direction vectors
    major_axis_vector = (major_axis[1] - major_axis[0]) / np.linalg.norm(
        major_axis[1] - major_axis[0]
    )
    minor_axis_vector = np.array(
        [-major_axis_vector[1], major_axis_vector[0]]
    )  # Perpendicular vector

    # Sample points along the major axis
    major_start, major_end = major_axis
    sampled_major_points = [
        (major_start * (1 - t) + major_end * t) for t in np.linspace(0, 1, num_points)
    ]

    # Shift the sampled points to be centered along the minor axis
    sampled_points = [
        point + (center - np.mean([major_start, major_end], axis=0))
        for point in sampled_major_points
    ]
    sampled_points = np.array(sampled_points)
    sampled_points = sampled_points[:, [1, 0]]

    return sampled_points


def get_major_axis_pts(mask, vis=False, output_path=None):
    # Ensure the mask is binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Get coordinates of the white pixels
    points = np.column_stack(np.where(binary_mask > 0))

    # Calculate the mean (center of mass)
    center = np.mean(points, axis=0)

    # Centralize the points (subtract the mean)
    centralized_points = points - center

    # Calculate the covariance matrix
    cov_matrix = np.cov(centralized_points.T)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Major and minor axis lengths (sqrt of eigenvalues)
    major_axis_length = 2 * np.sqrt(
        eigenvalues[0]
    )  # Multiply by 2 for the full axis length
    minor_axis_length = 2 * np.sqrt(eigenvalues[1])

    # Directions of the axes
    major_axis_vector = eigenvectors[:, 0]
    minor_axis_vector = eigenvectors[:, 1]

    # Visualize the result
    output = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Draw the center
    center = tuple(np.round(center).astype(int))
    # cv2.circle(output, center, 5, (0, 0, 255), -1)

    # Draw the major axis
    major_endpoint1 = center + np.round(
        major_axis_vector * major_axis_length / 2
    ).astype(int)
    major_endpoint2 = center - np.round(
        major_axis_vector * major_axis_length / 2
    ).astype(int)
    # cv2.line(output, tuple(major_endpoint2), tuple(major_endpoint1), (0, 255, 0), 2)

    # Draw the minor axis
    minor_endpoint1 = center + np.round(
        minor_axis_vector * minor_axis_length / 2
    ).astype(int)
    minor_endpoint2 = center - np.round(
        minor_axis_vector * minor_axis_length / 2
    ).astype(int)
    # cv2.line(output, tuple(minor_endpoint2), tuple(minor_endpoint1), (255, 0, 0), 2)

    if vis:
        plt.imshow(mask)
        plt.scatter(center[1], center[0], c="r")
        plt.scatter(major_endpoint1[1], major_endpoint1[0], c="g")
        plt.scatter(major_endpoint2[1], major_endpoint2[0], c="b")
        # plt.scatter(minor_endpoint1[1], minor_endpoint1[0], c='w')
        # plt.scatter(minor_endpoint2[1], minor_endpoint2[0], c='m')
        # plt.show()
        plt.savefig(f"{output_path}eigvec.png")
    major_axis_pts = np.array([center, major_endpoint1, major_endpoint2])
    major_axis_pts = major_axis_pts[:, [1, 0]]
    return major_axis_pts


def get_fg_bg_pts(mask, use_fg_pts, use_bg_pts, num_fg_pts=70, num_bg_pts=30):
    # Initialize input points and labels
    input_points = None
    input_labels = None
    # Sample foreground points
    evenly_distributed = True
    if use_fg_pts:
        # Get the coordinates of all non-zero pixels in the mask
        non_zero_coords = np.argwhere(mask)
        if evenly_distributed:
            step = max(1, len(non_zero_coords) // num_fg_pts)
            sampled_indices = non_zero_coords[::step]
        else:
            if non_zero_coords.shape[0] > num_fg_pts:
                indices = np.random.choice(
                    non_zero_coords.shape[0], num_fg_pts, replace=False
                )
                sampled_indices = non_zero_coords[indices]
        non_zero_points = sampled_indices[:, [1, 0]]
        non_zero_labels = np.ones(non_zero_points.shape[0], dtype=np.int32)

    # Set every other pixel in the mask that has a value of 0 to be a mask point with a label of 0
    even_dist_bg_pts = False
    if use_bg_pts:
        zero_coords = np.argwhere(mask == 0)
        if not even_dist_bg_pts:
            if zero_coords.shape[0] > num_bg_pts:
                indices = np.random.choice(
                    zero_coords.shape[0], num_bg_pts, replace=False
                )
                zero_coords = zero_coords[indices]
        else:
            step = max(1, len(zero_coords) // num_bg_pts)
            zero_coords = zero_coords[::step]
        zero_points = zero_coords[:, [1, 0]]
        zero_labels = np.zeros(zero_points.shape[0], dtype=np.int32)

    # Combine mask points and zero points
    if use_fg_pts and use_bg_pts:
        input_points = np.concatenate((non_zero_points, zero_points), axis=0)
        input_labels = np.concatenate((non_zero_labels, zero_labels), axis=0)
    elif use_bg_pts:
        input_points = zero_points
        input_labels = zero_labels
    elif use_fg_pts:
        input_points = non_zero_points
        input_labels = non_zero_labels

    # Convert input points and labels to numpy arrays
    # input_points = np.array(input_points, dtype=np.int32)
    # input_labels = np.array(input_labels, dtype=np.int32)
    return input_points, input_labels


def filter_mask(mask, output_path):
    from sklearn.cluster import DBSCAN

    # Get the coordinates of non-zero pixels (foreground)
    foreground_points = np.column_stack(np.where(mask == True))
    eps = 5  # Adjust based on the scale of your image
    min_samples = 10  # Adjust based on the density of the noise

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(foreground_points)
    # Get unique labels and their counts
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Initialize an empty mask for the cleaned image
    cleaned_mask = np.zeros_like(mask)

    max_label = unique_labels[np.argmax(counts)]
    cluster_indices = np.where(labels == max_label)
    cleaned_mask[
        foreground_points[cluster_indices, 0], foreground_points[cluster_indices, 1]
    ] = 255

    # Define minimum cluster size (number of pixels) to keep
    # min_cluster_size = 100  # Adjust based on your needs
    # # Iterate over clusters
    # for label, count in zip(unique_labels, counts):
    #     if label == -1:
    #         # Skip noise labeled by DBSCAN
    #         continue
    #     if count >= min_cluster_size:
    #         # Get indices of points in the cluster
    #         cluster_indices = np.where(labels == label)
    #         # Set these points to 255 in the cleaned mask
    #         cleaned_mask[foreground_points[cluster_indices, 0], foreground_points[cluster_indices, 1]] = 255
    # Save the cleaned mask image
    cv2.imwrite(f"{output_path}cleaned_mask.png", cleaned_mask.astype(int) * 255)
    return cleaned_mask.astype(bool)


def project_robot_links(img, urdf, joints, gripper_pos):
    """Project link pts into camera frame."""
    from autolab_core import RigidTransform, Point

    camera_to_world = RigidTransform.load("/home/inhand/data/config/cam2world.tf")
    intrinsics = np.load("/home/inhand/camera_calibration.npz")
    camera_matrix = intrinsics["camera_matrix"]
    dist_coeffs = intrinsics["dist_coeffs"]

    world_to_camera = camera_to_world.inverse()

    curr_cfg = YUMI_RIGHT_NO_POV_POSE.copy()
    for key, value in zip(YUMI_LEFT_JOINTS, joints):
        curr_cfg[key] = value
    curr_cfg["gripper_l_joint"] = gripper_pos
    urdf.update_cfg(curr_cfg)

    image_with_point = img.copy()
    gripper_links = [
        "gripper_r_finger_r",
        "gripper_r_finger_l",
        # "gripper_l_base",
        "gripper_l_finger_l",
        "gripper_l_finger_r",
    ]
    projected_links = {}
    print("Projecting joint points...")
    for link in YUMI_LINKS:
        transform = urdf.get_transform(frame_to=link, frame_from="base_link")
        link_in_world = transform[:3, 3].copy()
        if link in gripper_links:
            link_in_world += transform[:3, :3] @ np.array(
                [0, 0, 0.02]
            )  # Shift 2 cm in the z-axis
            # pass
        link_in_camera = world_to_camera * Point(link_in_world, frame="world")
        link_point = np.array(
            [[link_in_camera.x, link_in_camera.y, link_in_camera.z]]
        )  # 3D point as a 1x3 array
        print(f"Link {link} is pt {link_point} in camera frame")

        link_pixel, _ = cv2.projectPoints(
            link_point,
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            camera_matrix,
            dist_coeffs,
        )

        x, y = (int(link_pixel[0][0][0]), int(link_pixel[0][0][1]))
        if (
            x >= 0
            and y >= 0
            and x < image_with_point.shape[1]
            and y < image_with_point.shape[0]
        ):
            print(f"Drawing link {link} at {x}, {y}")
            projected_links[link] = (x, y)
            if link == "gripper_l_finger_r":
                color = (255, 0, 0)
            elif link == "gripper_l_finger_l":
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            image_with_point = cv2.circle(
                image_with_point,
                (x, y),
                radius=5,
                color=color,
                thickness=-1,
            )
    cv2.imwrite(f"/home/inhand/inhand/out_data/link_projected2.jpg", image_with_point)
    return np.array([x for x in projected_links.values()])


def predict_images(
    predictor,
    data_path,
    mask_files,
    image_files,
    use_fg_pts=False,
    use_bg_pts=False,
    use_bbox=True,
):
    # Iterate over each mask file
    for mask_file, image_file in zip(mask_files, image_files):
        image_name = os.path.basename(image_file)
        # Extract the number from the image name
        image_number = "".join(filter(str.isdigit, image_name))
        print(f"Image name: {image_name}, Number: {image_number}")
        mask = np.array(Image.open(mask_file))
        image = Image.open(image_file)
        image = np.array(image)  # .convert("RGB"))
        # Convert RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        input_points, input_labels = get_fg_bg_pts(mask, use_fg_pts, use_bg_pts)
        bbox = get_bbox(mask) if use_bbox else None

        plt.figure(figsize=(9, 6))
        plt.imshow(image)
        if use_bg_pts or use_fg_pts:
            show_points(input_points, input_labels, plt.gca())
        if use_bbox:
            show_box(bbox, plt.gca())
        plt.axis("on")
        plt.show()
        plt.savefig(f"{output_path}inputs.png")

        predictor.set_image(image)

        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=bbox,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        sam_mask = (masks[0]).astype(bool)
        masked_rgb = image.copy()
        masked_rgb[~sam_mask] = 0
        # show_masks(image, masks, scores, point_coords=input_points, input_labels=input_labels, borders=True)
        cv2.imwrite(f"{output_path}masked_imgs/{image_number}.png", masked_rgb)


def predict_video(
    predictor,
    mask_file,
    data_path,
    use_fg_pts,
    use_bg_pts,
    use_bbox,
):
    # TODO: need to refactor all of this code it's so ugly
    video_dir = os.path.join(data_path, "images")
    image_files = sorted(os.listdir(video_dir))
    image_files = [
        os.path.join(video_dir, image_files[i]) for i in range(0, len(image_files))
    ]
    # NOTE: REMEMBER TO CHANGE BACK!!!
    image = np.array(Image.open(image_files[7]))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    inference_state = predictor.init_state(video_path=video_dir)

    # use this if you want to redo what you're tracking
    # predictor.reset_state(inference_state)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = (
        1  # give a unique id to each object we interact with (it can be any integers)
    )

    use_mask = False
    if use_mask:
        mask = np.array(Image.open(mask_file))
        input_points, input_labels = get_fg_bg_pts(
            mask, use_fg_pts, use_bg_pts, num_fg_pts=7, num_bg_pts=0
        )
        # bbox = get_bbox(mask) if use_bbox else None
        bbox = get_obbox(mask) if use_bbox else None

    # def get_center_pts(data):
    #     ca = np.cov(data,y = None,rowvar = 0,bias = 1)
    #     v, vect = np.linalg.eig(ca)
    #     tvect = np.transpose(vect)
    #     #use the inverse of the eigenvectors as a rotation matrix and
    #     #rotate the points so they align with the x and y axes
    #     ar = np.dot(data,np.linalg.inv(tvect))
    #     # get the minimum and maximum x and y
    #     mina = np.min(ar,axis=0)
    #     maxa = np.max(ar,axis=0)
    #     diff = (maxa - mina)*0.5
    #     # the center is just half way between the min and max xy
    #     center = mina + diff
    #     first_third = center - diff/2
    #     second_third = center + diff/2
    #     #get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
    #     corners = np.array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],center+[diff[0],diff[1]],center+[-diff[0],diff[1]],center+[-diff[0],-diff[1]]])

    #     #use the the eigenvectors as a rotation matrix and
    #     #rotate the corners and the centerback
    #     corners = np.dot(corners,tvect)
    #     center = np.dot(center,tvect)

    #     plt.figure(figsize=(9, 6))
    #     plt.imshow(mask)
    #     plt.scatter(bbox[:, 1], bbox[:, 0], c='red', s=4) if data.shape[1] == 2 else None
    #     plt.plot(bbox[:,1], bbox[:,0],'-', c='red') if data.shape[1] == 2 else None
    #     plt.scatter([center[1]],[center[0]])
    #     # plt.plot([first_third[1],first_third[0]],[second_third[1],second_third[0]],'-')
    #     plt.plot([second_third[1],first_third[1]],[second_third[0],first_third[0]], '-')
    #     plt.plot(corners[:,1],corners[:,0],'-')
    #     plt.show()
    #     plt.savefig(f"{output_path}center_pts.png")
    #     center_pts = np.array([center, first_third, second_third])
    #     center_pts = center_pts[:, [1, 0]]
    #     return center_pts
    # # Calculate the major and minor axis lengths from the oriented bounding box
    # edge_lengths = np.linalg.norm(bbox[1] - bbox[0]), np.linalg.norm(bbox[2] - bbox[1])
    # major_axis_length = max(edge_lengths)
    # minor_axis_length = min(edge_lengths)
    # print(f"Major axis length: {major_axis_length}, Minor axis length: {minor_axis_length}")
    # # Calculate the center of the bounding box
    # center_x = np.mean(box[:, 0])
    # center_y = np.mean(box[:, 1])
    # print(f"Center of the bounding box: ({center_x}, {center_y})")

    use_major_axis = False
    if use_major_axis:
        input_points = get_major_axis_pts(mask, vis=True, output_path=data_path)
        breakpoint()
        newpt = input_points[2] + 1.65 * (input_points[2] - input_points[0]) / 3
        newpt = newpt.reshape(1, 2)
        input_points = np.append(input_points, newpt, axis=0)
        # input_points = sample_points_centered_major_axis(bbox, num_points=5)
        # input_points = input_points[1:-1]
        plt.imshow(mask)
        for pt in input_points:
            plt.scatter(pt[0], pt[1], c="r")
        # plt.show()
        plt.savefig(f"{data_path}majoraxis.png")
        input_labels = np.ones(input_points.shape[0], dtype=np.int32)

    use_project_robot_links = False
    if use_project_robot_links:
        from pathlib import Path
        import yourdfpy

        joint_angles = np.load(os.path.join(data_path, "joint_angles.npy"))
        gripper_positions = np.load(os.path.join(data_path, "gripper_positions.npy"))
        urdf_path = Path("/home/inhand/inhand/data/yumi_description/urdf/yumi.urdf")
        urdf = yourdfpy.URDF.load(urdf_path)
        input_points = project_robot_links(
            img=image,
            urdf=urdf,
            joints=joint_angles[7],
            gripper_pos=gripper_positions[7],
        )
        input_labels = np.ones(input_points.shape[0], dtype=np.int32)

    use_manual_pts = True
    if use_manual_pts:
        # left_clicks = []
        # right_clicks = []

        # def onclick(event):
        #     """Handles mouse click events."""
        #     if event.button == 1:  # Left mouse button
        #         left_clicks.append((int(event.xdata), int(event.ydata)))
        #         print(f"Left click at: ({int(event.xdata)}, {int(event.ydata)})")
        #     elif event.button == 3:  # Right mouse button
        #         right_clicks.append((int(event.xdata), int(event.ydata)))
        #         print(f"Right click at: ({int(event.xdata)}, {int(event.ydata)})")

        # # Set up the plot
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # ax.set_title("Left click to select points to segment, Right click to select points to NOT segment.\nClose the window when done.")

        # # Connect the click event to the handler
        # cid = fig.canvas.mpl_connect('button_press_event', onclick)

        # # Show the image
        # plt.show()

        # # Disconnect the event handler
        # fig.canvas.mpl_disconnect(cid)

        # # Print the collected points
        # print("Left click points:", left_clicks)
        # print("Right click points:", right_clicks)

        # @zehan update these two lists!
        # left_clicks = [(229, 303), (169, 961), (1198, 570), (1999, 631), (2073, 934)]
        # right_clicks = [(2235, 701), (2583, 764)]
        # scissors first
        # left_clicks = [(231, 331), (209, 1025), (1188, 570), (1996, 644), (1996, 910)]
        # right_clicks = [(2610, 378), (2371, 780), (1857, 1239)]
        # scissors second
        # left_clicks = [(215, 338), (221, 940), (1196, 578), (2022, 649), (2052, 940)]
        # right_clicks = [(2612, 724), (3133, 747)]
        # icecream second
        # left_clicks = [(230, 269), (1178, 618), (2051, 939)]
        # right_clicks = [(1919, 126), (1962, 502), (2194, 737)]
        # icecream third
        # left_clicks = [(236, 528), (1180, 619), (2025, 644), (2031, 923)]
        # right_clicks = [(2281, 654), (2519, 719), (2431, 891), (2222, 1204)]

        #jan_13_icecream_0
        left_clicks = [(193, 655), (1064, 634), (2080, 1105)]
        right_clicks = [(2344, 655), (2278, 939)]
        
        left_clicks = np.array(left_clicks)
        right_clicks = np.array(right_clicks)
        left_labels = np.zeros(left_clicks.shape[0], dtype=np.int32)
        right_labels = np.ones(right_clicks.shape[0], dtype=np.int32)
        #labels 1 (foreground point) or 0 (background point)
        input_labels = np.concatenate((left_labels, right_labels), axis=0)
        input_points = np.concatenate((left_clicks, right_clicks), axis=0)

    bbox = None

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=input_points,
        labels=input_labels,
        box=bbox,
    )

    sam2_mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
    filtered_mask = filter_mask(sam2_mask, data_path)
    filtered_mask_tensor = (
        torch.tensor(filtered_mask, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(image_files[ann_frame_idx]))
    if input_points is not None and input_labels is not None:
        show_points(input_points, input_labels, plt.gca())
    if bbox is not None:
        # show_box(bbox, plt.gca())
        pass
    show_mask_vid(
        (filtered_mask_tensor[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0]
    )
    plt.savefig(f"{data_path}inputs.png")

    # NOTE: seems that even though we clean the first frame this cleaned frame is not sent to sam2?
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for (
        out_frame_idx,
        out_obj_ids,
        filtered_mask_tensor,
    ) in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (filtered_mask_tensor[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    vis_frame_stride = 1
    plt.close("all")

    os.makedirs(f"{data_path}masked_imgs/", exist_ok=True)
    for out_frame_idx in range(0, len(image_files), vis_frame_stride):
        # plt.figure(figsize=(6, 4))
        # plt.title(f"frame {out_frame_idx}")
        curr_img = cv2.imread(image_files[out_frame_idx])
        # plt.imshow(curr_img)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            sam_mask = out_mask.squeeze(0).astype(bool)
            masked_rgb = curr_img
            masked_rgb[~sam_mask] = 0
            # Ensure the frame index has leading zeros based on the maximum number of digits in the total number of image files
            max_digits = len(str(len(image_files)))
            frame_idx_str = str(out_frame_idx).zfill(max_digits)
            cv2.imwrite(f"{data_path}masked_imgs/{frame_idx_str}.png", masked_rgb)
    # NOTE: to save a video
    # cd into the masked_imgs directory and run the following command:
    # ffmpeg -framerate 30 -i '%03d.png' -c:v libx264 -pix_fmt yuv420p output.mp4


def main(
    pred_vid: bool = True,
    use_fg_pts: bool = True,
    use_bg_pts: bool = False,
    use_bbox: bool = False,
):
    sam2_checkpoint = "/home/inhand/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # NOTE: @zehan want to update this!
    data_path = "/home/inhand/inhand/data/ice_cream_0/"
    # TODO: gotta clean this up
    video_dir = os.path.join(data_path, "images")
    # output_path = "home/inhand/inhand/out_data/"
    # todo change

    # SAM2 requires the images to be named in a specific format
    import re
    def rename_images(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".jpg"):
                match = re.search(r"\d+", filename)
                if match:
                    new_name = f"{int(match.group()):04d}.jpg"
                    os.rename(
                        os.path.join(directory, filename),
                        os.path.join(directory, new_name),
                    )
    rename_images(video_dir)

    masks = os.path.join(data_path, "masks")
    masked_imgs_path = os.path.join(data_path, "masked_imgs")
    os.makedirs(masked_imgs_path, exist_ok=True)

    # Get every mask from masks directory
    mask_files = sorted(os.listdir(masks))
    mask_files = [os.path.join(masks, mask_files[i]) for i in range(0, len(mask_files))]

    if pred_vid:
        from sam2.build_sam import build_sam2_video_predictor
        breakpoint()
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        predict_video(
            predictor,
            mask_files[0],
            data_path,
            use_fg_pts,
            use_bg_pts,
            use_bbox,
        )
    else:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        # Get every tenth image from image_dir
        image_files = sorted(os.listdir(video_dir))
        image_files = [
            os.path.join(video_dir, image_files[i]) for i in range(0, len(image_files), 10)
        ]
        predict_images(
            predictor,
            mask_files,
            image_files,
            use_fg_pts,
            use_bg_pts,
            use_bbox,
        )


if __name__ == "__main__":
    # TODO: refactor to include tyro
    tyro.cli(main)