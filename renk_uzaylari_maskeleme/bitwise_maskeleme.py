import cv2
import numpy as np

image = cv2.imread("images/image.jpg")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# mavi renk hsv aralığı ve maske oluşturma
lower_blue = np.array([90, 80, 50])
upper_blue = np.array([130, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue) # mavileri beyaz olarak diğerlerini siyah olarak işaretler

# maskenin tersi
inverse_mask = cv2.bitwise_not(blue_mask) # mavileri siyah ve diğerlerini beyaz olarak işaretler

# maskeyi gerçek görüntüye uygula
blue_result = cv2.bitwise_and(image, image, mask=blue_mask)
background_result = cv2.bitwise_and(image, image, mask=inverse_mask)

# iki sonucu tekrar birleştir
combined = cv2.bitwise_or(blue_result, background_result)

# XOR ile iki maske farkı
xor_mask = cv2.bitwise_xor(blue_mask, inverse_mask)

# sonuçları göster
cv2.imshow("orijinal", image)
cv2.imshow("blue_mask", blue_mask)
cv2.imshow("inverse_mask", inverse_mask)
cv2.imshow("Sadece mavi bolgeler", blue_result)
cv2.imshow("Mavi olmayan bolgeler", background_result)
cv2.imshow("tekrar birlestirilmis", combined)
cv2.imshow("xor maskesi", xor_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()