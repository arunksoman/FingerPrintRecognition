import os
import glob
import numpy as np
import cv2
import sys
from Enhance_FP.image_enhance import image_enhance

base_dir = os.path.abspath(os.path.dirname(__file__))
print(base_dir)
dataset_dir = os.path.join(base_dir, 'RawDataset')
img_files = glob.glob(dataset_dir + '\*.bmp')
print(img_files)
enhanced_dir = os.path.join(base_dir, 'enhanced')
if not os.path.exists(enhanced_dir):
    os.mkdir(enhanced_dir)

if __name__ == '__main__':
    for image in img_files:
        img_name = image.split(os.path.sep)[-1]
        img = cv2.imread(image)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        rows,cols = img.shape
        aspect_ratio = np.double(rows)/np.double(cols)
        new_rows = 350; # randomly selected number
        new_cols = new_rows/aspect_ratio
        img = cv2.resize(img,(np.int(new_cols),np.int(new_rows)))
        enhanced_img = image_enhance(img)
        print('saving the image {}'.format(img_name))
        cv2.imwrite(os.path.join(enhanced_dir, img_name), (255*enhanced_img))
