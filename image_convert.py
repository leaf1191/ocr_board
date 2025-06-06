import numpy as np
from PIL import Image
import cv2 as cv
from image_process import *

def pil_to_cv(pil_img):
    cv_img = np.array(pil_img)
    if cv_img.dtype == np.float32 or cv_img.dtype == np.float64:
        cv_img = (cv_img * 255).astype(np.uint8)
    cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2BGR)
    return cv_img

def cv_to_pil(cv_img):
    pil_img = Image.fromarray(cv_img)
    return pil_img

def resize_image(image, width=800):
    # width = 800 적절히 변환해보면 될 듯
    height = int((width / image.shape[1]) * image.shape[0])
    resized_image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
    return resized_image

def get_binary_image(image):
    image = convert_to_gray(image)
    image = apply_clahe(image)
    image = sharpen_image(image)
    image = apply_bilateral_filter(image)
    image = apply_threshold(image)
    return image

def remove_noise(image):
    image = apply_morphology(image)
    image = apply_dilation(image)
    image = remove_small_noise(image)
    return image

def process_image(image):
    if isinstance(image, Image.Image):
        cv_image = pil_to_cv(image)
    else:
        cv_image = image
    
    image = get_binary_image(cv_image)
    image = remove_noise(image)
    return image