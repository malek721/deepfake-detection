import os
import shutil
import glob
from sklearn.model_selection import train_test_split

original_dataset_dir = r'C:\Users\admin\Desktop\yapa zeka\dataset'
base_dir = r'C:\Users\admin\Desktop\yapa zeka\deepfake_split_dataset'


for split in ['train', 'val', 'test']:
    for category in ['real', 'fake']:
        os.makedirs(os.path.join(base_dir, split, category), exist_ok=True)


def process_category(category):

    all_images = glob.glob(os.path.join(original_dataset_dir, category, '**', '*.*'), recursive=True)

    print(f'{category} - Total images found: {len(all_images)}')

    train_imgs, temp_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    print(f'{category} - Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}')


    def copy_files(file_list, destination_folder):
        for path in file_list:
            subfolder = os.path.basename(os.path.dirname(path))
            filename = subfolder + '_' + os.path.basename(path)
            dest = os.path.join(base_dir, destination_folder, category, filename)
            shutil.copy(path, dest)

    copy_files(train_imgs, 'train')
    copy_files(val_imgs, 'val')
    copy_files(test_imgs, 'test')

process_category('real')
process_category('fake')
