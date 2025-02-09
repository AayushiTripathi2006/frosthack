import os
import cv2
import imgaug.augmenters as iaa

def augment_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            augmenter = iaa.Fliplr(1.0)
            augmented_img = augmenter(image=img)

            output_path = os.path.join(output_folder, f'aug_{img_name}')
            cv2.imwrite(output_path, augmented_img)
            print(f"Augmented and saved: {output_path}")  # Console output for tracking

augment_images('C:/Users/amiru/OneDrive/Documents/Desktop/sdc/sdc_apc/data_ogmentation/datasets/images',
               'C:/Users/amiru/OneDrive/Documents/Desktop/sdc/sdc_apc/data_ogmentation/datasets/augmented_images')

