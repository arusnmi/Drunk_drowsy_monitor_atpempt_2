from PIL import Image
import os 



input_dir='tren_resize/'


output_dir='tren_greay/'


os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg','.jpeg','.png')):
        img=Image.open(os.path.join(input_dir,filename))
        grey_image=img.convert('L')
        grey_image.save(os.path.join(output_dir,filename))
