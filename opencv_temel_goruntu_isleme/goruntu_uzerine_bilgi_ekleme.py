import cv2

image = cv2.imread("images/bird.jpg")

# orijinal görüntüyü korumak için kopya oluştur
result = image.copy()
heigth, width = result.shape[:2]

# görüntünün üzerine başlık ekle
cv2.putText(result, "OpenCV Eğitimi", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# görüntünün boyut bilgilerini oluştur ve ekle
size_text = f"Boyut: {width}x{heigth}"
cv2.putText(result, size_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

# merkez noktasına açıklama ekle
center = (width // 2, heigth // 2)
cv2.circle(result, center, 6, (0,0,255), -1)
cv2.putText(result, "Merkez", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0 ,255), 2)

# görüntünün kanal bilgisi
channel_text = f"Kanal: {image.shape[2]}"
cv2.putText(result, channel_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0 ,255), 2)

# sonucu göster
cv2.imshow("Bilgi eklenmis goruntu", result)
cv2.waitKey(0)
cv2.destroyAllWindows()