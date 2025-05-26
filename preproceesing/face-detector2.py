import os
import cv2
from mtcnn import MTCNN

main_folder = r"C:\Users\admin\Desktop\yapa zeka\processed_original"


detector = MTCNN()


for root, dirs, files in os.walk(main_folder):
    for filename in files:
        image_path = os.path.join(root, filename)

        try:
            image = cv2.imread(image_path)

            if image is None:
                print(f"fotoğraf yükenmedi {filename}")
                os.remove(image_path)
                continue

            faces = detector.detect_faces(image)

            if not faces:
                print(f"silindi {filename} - yüz içermiyor.")
                os.remove(image_path)
            else:
                print(f"kayıtedildi {filename} -yüz içeriyor.")

        except Exception as e:
            print(f"️hata oldu {filename}: {e}")
