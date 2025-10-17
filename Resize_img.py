import os
from PIL import Image


input_dir='tren/'


output_dir='tren_resize/'


target_size=(227,227)



os.makedirs(output_dir, exist_ok=True)


for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg','.jpeg','.png')):
        img=Image.open(os.path.join(input_dir,filename))
        img_resized=img.resize(target_size)
        img_resized.save(os.path.join(output_dir,filename))