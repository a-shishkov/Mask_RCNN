import os

folder = 'C:\\Users\\nsakn\OneDrive\Documents\Result\Mask_RCNN\datasets\car_damage\\val'

i = 0
train_last = 80
for filename in os.listdir(folder):
    if filename.startswith('image'):
        i += 1
        src = f'{folder}/{filename}'
        dst = f'{folder}/{train_last + i}.jpg'
        os.rename(src, dst)
        