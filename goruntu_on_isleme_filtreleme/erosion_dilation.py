import cv2

gray = cv2.imread("images/morfoloji.jpg", cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Kernel oluştur
# MORPH_RECT: dikdörtgen kernel, boyu 5x5
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# erosion
# iterarions: erezyon işlemi 1 kere uygulanır
eroded = cv2.erode(binary, kernel, iterations=1)
eroded2 = cv2.erode(binary, kernel, iterations=2)

# dilation
dilated = cv2.dilate(binary, kernel, iterations=1)
dilated2 = cv2.dilate(binary, kernel, iterations=2)

# sonuçları göster
cv2.imshow("binary", binary)
cv2.imshow("eroded", eroded)
cv2.imshow("eroded2", eroded2)
cv2.imshow("dilated", dilated)
cv2.imshow("dilated2", dilated2)

cv2.waitKey(0)
cv2.destroyAllWindows()