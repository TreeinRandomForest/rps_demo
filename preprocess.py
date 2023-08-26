import cv2
import os
import glob
from PIL import Image

def canny(fname): 
    img = cv2.imread(fname) 

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
 
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=150) # Canny Edge Detection
    
    #return img, edges
    return edges

def preprocess(in_folder, 
               out_folder, 
               transform):
    
    if not os.path.exists(in_folder):
        raise ValueError(f'in_folder: {in_folder} does not exist')

    if os.path.exists(out_folder):
        raise ValueError(f'out_folder: {out_folder} already exists')
    
    os.makedirs(out_folder)
    for subdir in os.listdir(in_folder): #this is brittle - expects only labels here
        os.makedirs(f'{out_folder}{os.sep}{subdir}')

    in_files = glob.glob(f'{in_folder}/*/*')
    out_files = []
    for f in in_files:
        fname = f.split(os.sep)
        fname[-3] = out_folder.split(os.sep)[-1]
        out_fname = os.sep.join(fname)

        out_img = transform(f)
        Image.fromarray(out_img).save(out_fname)

    '''
    for subfolder in []:
        for file in :
            img = transform(file)

            img.save(out_folder, subfolder, file)
            '''
