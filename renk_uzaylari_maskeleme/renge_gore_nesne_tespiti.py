import cv2
import numpy as np

image = cv2.imread("images/image.jpg")

if image is None:
    raise FileNotFoundError("Goruntu bulunamadi")

result = image.copy()

# görüntüyü hsv renk uzayına dönüştür
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# bir önceki derste trackbar ile bulunan mavi renk aralığını kullan
lower_blue = np.array([7, 144, 178]) # [90, 80, 50]
upper_blue = np.array([19, 255, 255]) # [130, 255, 255]

# mavi bölgeler için maske oluştur
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# maskedeki küçük gürültüleri azalt
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# maskedeki nesnelerin konturlarını bul
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

detected_count = 0

for contour in contours:

    # çok küçük bölgeleri nesne olarak kabul etme
    area = cv2.contourArea(contour)

    if area < 500:
        continue

    # nesnenin çevresine kutu çiz
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # nesnenin merkez noktasını hesapla
    center_x = x + w // 2
    center_y = y + h // 2
    cv2.circle(result, (center_x, center_y), 5, (0 , 0, 255), -1)

    detected_count += 1

# tespit edilen nesne sayısını görüntüye ekle
cv2.putText(result, f"Nesne Sayisi: {detected_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# sonuçları göster
cv2.imshow("Renk Maskesi", mask)
cv2.imshow("Renge Gore Nesne Tespiti", result)

# tespit sonucunu kaydet
cv2.imwrite("outputs/renk_nesne_tespiti.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()