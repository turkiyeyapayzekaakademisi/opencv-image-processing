import cv2

image = cv2.imread("images/salt_pepper.png")

# 3: 3x3 lük kernel
median_small = cv2.medianBlur(image, 3)

# 5 ve 9 luk kernel boyutu ile filtre oluştur
median_medium = cv2.medianBlur(image, 5)
median_large = cv2.medianBlur(image, 9)

# sonuçları göster
cv2.imshow("orijinal", image)
cv2.imshow("Median 3x3", median_small)
cv2.imshow("Median 5x5", median_medium)
cv2.imshow("Median 9x9", median_large)

# medium filtreyi kaydet
cv2.imwrite("outputs/median_blur.jpg", median_medium)

cv2.waitKey(0)
cv2.destroyAllWindows()