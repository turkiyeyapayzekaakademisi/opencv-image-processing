import cv2

image = cv2.imread("images/bird.jpg")

# görüntünün yükseklik ve genişlik değerlerini al
height, width = image.shape[:2]

# görüntüyü yatı boyutuna küçült
resized = cv2.resize(image, (width // 2, height // 2))

# görüntünün orta bölümünü kırpmak için sınırları belirle
x1, x2 = width // 4, 3 * width // 4 # 1-3. çeyrekler arası
y1, y2 = height // 4, 3 * height // 4 # 1-3. çeyrekler arası

# belirlenen alanı kırp
cropped = image[y1:y2, x1:x2]

# sonuçları göster
cv2.imshow("Orijinal", image)
cv2.imshow("Boyutlandirilmis", resized)
cv2.imshow("Cropped", cropped)

# kırpılan görüntüyü kaydet
cv2.imwrite("outputs/cropped.jpg", cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()