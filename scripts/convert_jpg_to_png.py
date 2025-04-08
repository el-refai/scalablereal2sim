import os

from PIL import Image


def convert_jpg_to_png(source_folder, target_folder):
    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Loop over files in the source folder
    for filename in os.listdir(source_folder):
        if filename.lower().endswith((".jpg", ".jpeg")):
            source_path = os.path.join(source_folder, filename)
            # Open image
            img = Image.open(source_path)
            # Create new file name with png extension
            new_filename = os.path.splitext(filename)[0] + ".png"
            target_path = os.path.join(target_folder, new_filename)
            # Save image as PNG
            img.save(target_path, "PNG")
            # print(f"Converted {filename} to {new_filename}")


# Example usage:
# source_folder = '/home/xi/scalable-real2sim/data/cup/bundlesdf/rgb'
# target_folder = '/home/xi/scalable-real2sim/data/cup/bundlesdf/rgb_png'
convert_jpg_to_png(source_folder, target_folder)
