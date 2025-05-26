import cv2
import os
import numpy as np

input_folder = r'C:\Users\admin\Desktop\yapa zeka\train_frams'
output_folder = r'C:\Users\admin\Desktop\yapa zeka\resized_frams'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

target_size = 256


def resize_with_padding(image, target_size):
    h, w = image.shape[:2]

    if h > w:
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        new_w = target_size
        new_h = int(h * target_size / w)

    resized_img = cv2.resize(image, (new_w, new_h))

    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    final_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return final_img


def process_images(input_folder, output_folder, target_size):

    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)

        if os.path.isdir(subfolder_path):

            output_subfolder = os.path.join(output_folder, subfolder)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)


            for filename in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, filename)

                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):

                    img = cv2.imread(image_path)

                    if img is not None:

                        resized_img = resize_with_padding(img, target_size)

                        output_image_path = os.path.join(output_subfolder, filename)
                        cv2.imwrite(output_image_path, resized_img)
                        print(f'{filename} dosyası dönüştürüldü ve {output_image_path} yolunda kaydedildi.')


process_images(input_folder, output_folder, target_size)
