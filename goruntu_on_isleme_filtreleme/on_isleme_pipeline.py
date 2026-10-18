import cv2

image = cv2.imread("images/image.png")

if image is None:
    raise FileNotFoundError("goruntu bulunamadi")

# renk bilgisini kaldırarak tek kanallı gri görüntü oluştur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gürültü azaltma
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# adaptive threshold
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# küçük beyaz gürültüleri temizlemek için erosion + dilation
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

# küçük siyah boşlukları kapatmak için closing uygula -> dilation + erosion
processed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

# pipeline aşamalarını göster
cv2.imshow("1 - Orijinal", image)
cv2.imshow("2 - Gri Seviye", gray)
cv2.imshow("3 - Gaussian blur", blurred)
cv2.imshow("4 - Adaptive Threshold", binary)
cv2.imshow("5 - Final", processed)

# final sonucunu kaydet
cv2.imwrite("outputs/final_on_isleme.png", processed)

cv2.waitKey(0)
cv2.destroyAllWindows()