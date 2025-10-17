import os
import random
import shutil

random.seed(2095)

input_dir='tren_greay/'


output_dir='dataset'


os.makedirs(output_dir, exist_ok=True)




train_ratio=0.7
val_ratio=0.15
test_ration=0.15


assert train_ratio + val_ratio + test_ration == 1.0, "Ratios must sum to 1.0"

images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]

random.shuffle(images)

total_images = len(images)
train_images= int(total_images * train_ratio)
val_images= int(total_images * val_ratio)

train_files = images[:train_images]
val_files = images[train_images:train_images + val_images]
test_files = images[train_images + val_images:]

for folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir,folder), exist_ok=True)


def copy_files(file_list, dest_folder):
    for file_name in file_list:
        src_path = os.path.join(input_dir, file_name)
        dest_path = os.path.join(output_dir,dest_folder, file_name)
        shutil.copy2(src_path, dest_path)

copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')


