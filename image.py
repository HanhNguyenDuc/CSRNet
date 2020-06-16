import os
import random
import numpy as np 
import h5py 
from PIL import ImageStat, Image, ImageFilter, ImageDraw
import cv2
from typing import Optional, List


def load_data(img_path: str, train: Optional[bool]=True) -> List[Any]:
    output_path = img_path.replace('inputs', 'outputs')

    # Read image by PIL.Image will return valid form for torch.conv2d
    input_img = Image.open(img_path).convert('RGB')
    
    # Output read by cv2 because it's just 2 dims ndarray, so convert it to PIL.Image
    # is possible, without transpose
    output_img = cv2.imread(output_path, 0)
    
    # Reshape output image to suit model's output
    # but then we mul it by 64 because the decresing size of image
    old_shape = output_img.shape
    output_img = cv2.resize(
        output_img, 
        (output_img.shape[1] // 8, output_img.shape[0] // 8), 
        interpolation=cv2.INTER_CUBIC
    ) / 255 / 255 * 64

    return input_img, output_img

