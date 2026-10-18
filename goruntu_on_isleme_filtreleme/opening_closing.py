import cv2

gray = cv2.imread("images/morfoloji.jpg", cv2.IMREAD_GRAYSCALE)

# binary görüntü oluştur
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# kernel tanımla
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5 ,5))

# Opening (erosion + dilation): küçük beyaz gürültüleri temizle
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

# Closing (dilation + erosion): küçük siyah gürültüleri temizle
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

# önce opening ardından closing
cleaned = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

# sonuçları göster
cv2.imshow("binary", binary)
cv2.imshow("opening", opening)
cv2.imshow("closing", closing)
cv2.imshow("cleaned", cleaned)

cv2.waitKey(0)
cv2.destroyAllWindows()