'''
Created on 2017-02-11

@author: RenaiC
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import numpy as np
import scipy
from keras.preprocessing.image import load_img, img_to_array, array_to_img, list_pictures

def performance_evaluation(this_type,rst_type,Top_K):
    cnt = 0
    MRRK = 0
    
    for i,rst in enumerate(rst_type):
        if rst == this_type:
            cnt = cnt + 1
            MRRK = MRRK + 1./(i+1.)
    RK = cnt/50
    PK = cnt/len(rst_type)
    F1K = 2*PK*RK/(RK+PK)
    MRRK =MRRK/cnt
    
    return PK,RK,F1K,MRRK
        
    

def resize_imgs(input_dir, output_dir, target_width, target_height, quality=90, verbose=1):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("starting....")
        print("Collecting data from %s " % input_dir)
        for subdir in os.listdir(input_dir):
            input_subdir = os.path.join(input_dir, subdir)
            output_subdir = os.path.join(output_dir, subdir)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            if os.path.exists(output_subdir):
                for img_path in list_pictures(input_subdir):
                    try:
                        if verbose > 0:
                            print("Resizing file : %s " % img_path)

                        img = load_img(img_path)
                        zoom_factor = min(float(target_width) / img.width, float(target_height) / img.height)

                        img = img_to_array(img)

                        img = scipy.ndimage.interpolation.zoom(img, zoom=(1., zoom_factor, zoom_factor)) # not an error here

                        (_, height, width) = img.shape
                        pad_h_before = int(np.ceil(float(target_height - height) / 2))
                        pad_h_after = (target_height - height) / 2
                        pad_w_before = int(np.ceil(float(target_width - width) / 2))
                        pad_w_after = (target_width - width) / 2
                        img = np.pad(img, ((0, 0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)), mode='edge')

                        img = array_to_img(img)

                        _, fname_ext = os.path.split(img_path)
                        out_file = os.path.join(output_dir, subdir, fname_ext)
                        img.save(out_file, img.format, quality=quality)
                        img.close()
                    except Exception, e:
                        print("Error resize file : %s - %s " % (subdir, img_path))

    except Exception, e:
        print("Error, check Input directory etc : ", e)
        sys.exit(1)
    return output_dir

def resize_img():
    import PIL
    from PIL import Image
    baseheight = 560
    img = Image.open('fullsized_image.jpg')
    hpercent = (baseheight / float(img.size[1]))
    wsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
    img.save('resized_image.jpg')
    
if __name__ == "__main__": #resize all images one time
        src_path = r'H:\EclipseWorkspace\StatLearn\src\stat\project4\data'
        dst_path =r'H:\EclipseWorkspace\StatLearn\src\stat\project4\resized_data'
        resize_imgs(src_path, dst_path, 224, 224)
