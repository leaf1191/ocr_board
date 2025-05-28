import cv2 as cv
import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize
from skimage.util import invert

def convert_to_gray(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray_image

def apply_bilateral_filter(image, d=9, sigmaColor=50, sigmaSpace=75):
    filtered = cv.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return filtered

def apply_threshold(image, threshold=127, max_value=255, threshold_type=cv.THRESH_BINARY):
    _, binary = cv.threshold(image, threshold, max_value, threshold_type)
    return binary

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    result = clahe.apply(image)
    return result

def apply_morphology(image, kernel_size=(3, 3), iterations=1):
    inverted = cv.bitwise_not(image)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    image = cv.morphologyEx(inverted, cv.MORPH_CLOSE, kernel, iterations)
    return cv.bitwise_not(image)

def apply_dilation(image, kernel_size=(3, 3), iterations=1):
    inverted = cv.bitwise_not(image)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    dilated = cv.dilate(inverted, kernel, iterations=iterations)
    return cv.bitwise_not(dilated)

def sharpen_image(image, strength=1.0):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32) * strength
    kernel[1, 1] += 1.0
    sharpened = cv.filter2D(image, -1, kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def remove_small_noise(binary_image, min_size=20, connectivity=8):
    binary_image = cv.bitwise_not(binary_image)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        binary_image, connectivity=connectivity, ltype=cv.CV_32S)
    
    output = np.zeros_like(binary_image)
    
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        if area >= min_size:
            output[labels == i] = 255
    
    return cv.bitwise_not(output)

def skeletonize_skimage(binary_image, method='lee'):
    binary = cv.bitwise_not(binary_image)
    binary = (binary > 0).astype('uint8')
    skeleton = sk_skeletonize(binary, method=method)
    return cv.bitwise_not((skeleton * 255).astype('uint8'))

def adjust_brightness(image, factor=1.0):
    img_float = image.astype(np.float32)
    adjusted = np.clip(img_float * factor, 0, 255)
    return adjusted.astype(np.uint8)