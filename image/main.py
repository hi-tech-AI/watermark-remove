# import cv2

# def process(img):
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_blur = cv2.GaussianBlur(img_gray, (1, 1), 0)
#     img_canny = cv2.Canny(img_blur, 210, 190)
#     img_dilate = cv2.dilate(img_canny, None, iterations=1)
#     return cv2.erode(img_dilate, None, iterations=1)

# def get_watermark(img):
#     contours, _ = cv2.findContours(process(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     img.fill(255)
#     for cnt in contours:
#         if cv2.contourArea(cnt) > 100:
#             cv2.drawContours(img, [cnt], -1, 0, -1)

# img = cv2.imread("image1.png")
# get_watermark(img)
# cv2.imwrite('output.png', img)

import cv2
import numpy as np

def compare_n_img(array_of_img_paths):
    img_float_array = []
    for path in array_of_img_paths:
        img_float_array.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    additionF = sum(img_float_array) / len(img_float_array)
    addition = additionF.astype('uint8')
    return addition

# Load the watermarked images
img_paths = ['image3.jpeg']
composite_img = compare_n_img(img_paths)

# Apply edge detection
edges = cv2.Canny(composite_img, 100, 200)

# Perform dilation and erosion
kernel = np.ones((5,5),np.uint8)
mask = cv2.dilate(edges, kernel, iterations=1)
mask = cv2.erode(mask, kernel, iterations=1)

# Filter out small contours
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(mask)
for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        cv2.drawContours(mask, [cnt], 0, 255, -1)

# The mask is now ready to be used
cv2.imwrite('sample.png', mask)