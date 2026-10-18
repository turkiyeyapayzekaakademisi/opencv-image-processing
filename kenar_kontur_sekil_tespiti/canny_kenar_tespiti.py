import cv2

image = cv2.imread("images/sekiller.png")

# gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# gürültüyü azalt
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# kenar tespiti
# 50: alt eşik zayıf kenarların değerlendirilmesi için
# 150: üst eşik, güçlü kenarların belirlenmesi
edges = cv2.Canny(blurred, 75, 225)

# sonuçları göster
cv2.imshow("orijinal", image)
cv2.imshow("gri seviye", gray)
cv2.imshow("Canny", edges)

# kenar görüntüsünü kaydet
cv2.imwrite("outputs/canny_kenarlari.png", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()