import cv2 as cv
import numpy as np

def convert_to_gray(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray_image

def apply_bilateral_filter(image, d=9, sigmaColor=50, sigmaSpace=75):
    filtered = cv.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return filtered

def sobel_magnitude(image, ksize=3):
    sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=ksize)
    sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return magnitude

def apply_adaptive_threshold(image, max_value=255, method=cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                           threshold_type=cv.THRESH_BINARY, block_size=11, C=2):
    return cv.adaptiveThreshold(
        image, max_value, method, 
        threshold_type, block_size, C
    )

def apply_canny(image, threshold1=150, threshold2=200, aperture_size=3, L2gradient=False):
    edges = cv.Canny(image, threshold1, threshold2, 
                    apertureSize=aperture_size, 
                    L2gradient=L2gradient)
    return edges

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    result = clahe.apply(image)
    return result

def apply_morphology(image, kernel_size=(3, 3), iterations=2):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations)
    return image