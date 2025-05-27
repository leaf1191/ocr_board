import cv2 as cv
from image_process import *
def getImage():
    image = cv.imread('demo_photos/20250521_165030.jpg')
    image = resize_image(image)
    return image

def resize_image(image, width=800):
    height = int((width / image.shape[1]) * image.shape[0])
    resized_image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
    return resized_image

def process_image(image):
    image = convert_to_gray(image)
    image = apply_clahe(image)
    image = sharpen_image(image)
    image = apply_bilateral_filter(image)
    image = apply_threshold(image)
    image = apply_morphology(image)
    image = apply_dilation(image)
    image = remove_small_noise(image)
    return image

if __name__ == "__main__":
    image = getImage()
    image = process_image(image)

    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()