import cv2 as cv
from image_process import *
def getImage():
    image = cv.imread('demo_photos/20250521_164924.jpg')
    image = resize_image(image)
    return image

def resize_image(image, width=800):
    height = int((width / image.shape[1]) * image.shape[0])
    resized_image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
    return resized_image

if __name__ == "__main__":
    image = getImage()
    image = convert_to_gray(image)
    image = apply_gaussian_blur(image)
    image = apply_clahe(image)
    image = apply_inpainting(image)
    image = sobel_magnitude(image)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()