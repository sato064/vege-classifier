from rembg import remove
import cv2

img = cv2.imread('./ccb2.png')
rmvd_img = remove(img)
cv2.imwrite('./rmvd.png', rmvd_img)
