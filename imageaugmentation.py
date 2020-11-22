import os, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import librys

def generateImages(augmentation, value, cval=0.0):
    train_dir = librys.pathDir().loadDir()[0]
    # idx_label = np.random.randint(low=0, high=len(os.listdir(path=train_dir)))
    idx_label = 0
    idx_img = np.random.randint(low=0, high=len(os.listdir(path=train_dir+os.listdir(path=train_dir)[idx_label])))
    img_path = train_dir+os.listdir(path=train_dir)[idx_label]+'/'+os.listdir(path=train_dir+os.listdir(path=train_dir)[idx_label])[idx_img]
    img = imread(img_path)
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(25,5))

    if augmentation.lower() == "rotation_range":
        try:
            data_gen = ImageDataGenerator(rotation_range=value)
        except:
            return ValueError
    elif augmentation.lower() == "width_shift_range":
        try:
            data_gen = ImageDataGenerator(width_shift_range=value)
        except:
            return ValueError
    elif augmentation.lower() == "height_shift_range":
        try:
            data_gen = ImageDataGenerator(height_shift_range=value)
        except:
            return ValueError
    elif augmentation.lower() == "shear_range":
        try:
            data_gen = ImageDataGenerator(shear_range=value)
        except:
            return ValueError
    elif augmentation.lower() == "brightness_range":
        try:
            data_gen = ImageDataGenerator(brightness_range=value)
        except:
            return ValueError
    elif augmentation.lower() == "zoom_range":
        try:
            data_gen = ImageDataGenerator(zoom_range=value)
        except:
            return ValueError
    elif augmentation.lower() == "channel_shift_range":
        try:
            data_gen = ImageDataGenerator(channel_shift_range=value)
        except:
            return ValueError
    elif augmentation.lower() == "horizontal_flip":
        try:
            data_gen = ImageDataGenerator(horizontal_flip=value)
        except:
            return ValueError
    elif augmentation.lower() == "vertical_flip":
        try:
            data_gen = ImageDataGenerator(vertical_flip=value)
        except:
            return ValueError
    elif augmentation.lower() == "fill_mode":
        try:
            data_gen = ImageDataGenerator(width_shift_range=0.4, fill_mode=value, cval=cval)
        except:
            return ValueError
    else:
        return ValueError

    images_iter = data_gen.flow(np.expand_dims(image.img_to_array(img), axis=0))

    for col in range(5):
        ax[col].imshow(images_iter.next()[0].astype('int'))
        ax[col].axis('off')

    plt.suptitle(augmentation+" = "+str(value), fontsize=26, fontweight='bold')
    plt.show()