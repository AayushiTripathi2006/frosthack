import cv2
import os
import numpy as np

def resize_and_normalize_images(input_folder, output_folder, target_size=(128, 128)):
    """
    Resizes and normalizes images to a standard size and saves them to the output folder.
    
    Args:
        input_folder (str): Path to the folder containing raw images.
        output_folder (str): Path to the folder to save preprocessed images.
        target_size (tuple): Target image size (width, height).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            # Resize the image to target size
            img_resized = cv2.resize(img, target_size)

            # Normalize pixel values to the range [0, 1]
            img_normalized = img_resized / 255.0

            # Save the preprocessed image
            output_path = os.path.join(output_folder, img_name)
            cv2.imwrite(output_path, (img_normalized * 255).astype(np.uint8))

    print(f"Preprocessed images saved in: {output_folder}")

# Example usage
resize_and_normalize_images("../datasets/raw_footage", "../datasets/preprocessed_images")
