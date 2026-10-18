import cv2

# görüntüyü dosyadan oku
image = cv2.imread("images/bird.jpg")

# görüntünün doğru okunup okunmadığını kontrol etme
if image is None:
    raise FileNotFoundError("Goruntu bulunamadi")

# görüntüyü ekranda göster
cv2.imshow("Ornek Goruntu", image)

# görüntünün bir kopyasını outputs klasörüne kaydet
cv2.imwrite("outputs/kopya.jpg", image)

# bir tuşa basılmasını bekle ve pencereleri kapat
cv2.waitKey(0)
cv2.destroyAllWindows()