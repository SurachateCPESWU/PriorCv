import cv2

imgOrigin = cv2.imread("8.png")
imgOutput = imgOrigin
imgGray = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2GRAY)

ret, binImg = cv2.threshold(imgGray, 190, 255, cv2.THRESH_BINARY_INV)


horizontal_img = binImg.copy()
vertical_img = binImg.copy()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60,1))
horizontal_img = cv2.erode(horizontal_img, kernel, iterations=1)
horizontal_img = cv2.dilate(horizontal_img, kernel, iterations=1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
vertical_img = cv2.erode(vertical_img, kernel, iterations=1)
vertical_img = cv2.dilate(vertical_img, kernel, iterations=1)

mask_img = horizontal_img + vertical_img

output = binImg - mask_img

cv2.imshow("remove",output)
cv2.waitKey(0)
cv2.destroyAllWindows()