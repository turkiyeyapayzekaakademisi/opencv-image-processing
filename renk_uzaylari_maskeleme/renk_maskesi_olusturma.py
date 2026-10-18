import cv2
import numpy as np

image = cv2.imread("images/image.jpg")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# mavi renk hsv aralığı
lower_blue = np.array([90, 80, 50])
upper_blue = np.array([130, 255, 255])

# belirlenen renk aralığında pikselleri maskele
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# maskede ki beyaz piksel sayısı
white_pixels = cv2.countNonZero(mask)
print(f"Maskede bulunan piksel sayısı: {white_pixels}")

# sonuçları
cv2.imshow("Orijinal goruntu", image)
cv2.imshow("Mavi renk maskesi", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()