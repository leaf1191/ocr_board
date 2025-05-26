import cv2 as cv
import numpy as np

def convert_to_gray(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray_image

def apply_gaussian_blur(image, ksize=(5, 5), sigmaX=0):
    blurred = cv.GaussianBlur(image, ksize=ksize, sigmaX=sigmaX)
    return blurred

def sobel_magnitude(image, ksize=3):
    sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=ksize)
    sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return magnitude

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    result = clahe.apply(image)
    return result