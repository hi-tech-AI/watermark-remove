import cv2

def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_canny = cv2.Canny(img_blur, 161, 54)
    img_dilate = cv2.dilate(img_canny, None, iterations=1)
    return cv2.erode(img_dilate, None, iterations=1)

def get_watermark(img):
    contours, _ = cv2.findContours(process(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img.fill(255)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            cv2.drawContours(img, [cnt], -1, 0, -1)

def mask_generator():
    img = cv2.imread("image.jpg")
    get_watermark(img)
    cv2.imshow("Watermark", img)
    cv2.waitKey(0)
    cv2.imwrite("mask.jpg", img)