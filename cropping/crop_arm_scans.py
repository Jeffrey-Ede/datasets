from scipy.misc import imread
from PIL import Image
import os

data_loc = "E:/ARM_scans/"
save_loc = "E:/ARM_scans-crops/"

files = os.listdir(data_loc)
num_files = len(files)

train_files = files[:int(0.75*num_files)]
val_files = files[int(0.75*num_files):int(0.85*num_files)]
test_files = files[int(0.85*num_files):]

num_train_files = len(train_files)
num_val_files = len(val_files)
num_test_files = len(test_files)

train_save_loc = save_loc+"train/train"
val_save_loc = save_loc+"val/val"
test_save_loc = save_loc+"test/test"

counter = 1
for i, file in enumerate(train_files):
    print("Train file {} of {}".format(i, num_train_files))
    try:
        img = imread(data_loc+file, mode="F")

        if img.shape[0] >= 512 and img.shape[1] >= 512:
            for i in range(0, img.shape[0]-512+1, 512):
                for j in range(0, img.shape[1]-512+1, 512):
                    Image.fromarray(img[i:(i+512), j:(j+512)]).save( train_save_loc+str(counter)+".tif" )
                    counter += 1
    except:
        pass

counter = 1
for i, file in enumerate(val_files):
    print("Val file {} of {}".format(i, num_val_files))
    try:
        img = imread(data_loc+file, mode="F")

        if img.shape[0] >= 512 and img.shape[1] >= 512:
            for i in range(0, img.shape[0]-512+1, 512):
                for j in range(0, img.shape[1]-512+1, 512):
                    Image.fromarray(img[i:(i+512), j:(j+512)]).save( val_save_loc+str(counter)+".tif" )
                    counter += 1
    except:
        pass

counter = 1
for i, file in enumerate(test_files):
    print("Test file {} of {}".format(i, num_test_files))
    try:
        img = imread(data_loc+file, mode="F")

        if img.shape[0] >= 512 and img.shape[1] >= 512:
            for i in range(0, img.shape[0]-512+1, 512):
                for j in range(0, img.shape[1]-512+1, 512):
                    Image.fromarray(img[i:(i+512), j:(j+512)]).save( test_save_loc+str(counter)+".tif" )
                    counter += 1
    except:
        pass