import cv2
import numpy as np

# trackbar için boş callback fonksiyonu
def nothing(value):
    pass

image = cv2.imread("images/image.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# trackbar penceresi oluştur
cv2.namedWindow("HSV Ayarlari")

# alt hsv değerleri
cv2.createTrackbar("H Min", "HSV Ayarlari", 0, 179, nothing)
cv2.createTrackbar("S Min", "HSV Ayarlari", 0, 255, nothing)
cv2.createTrackbar("V Min", "HSV Ayarlari", 0, 255, nothing)

# üst hsv değerleri
cv2.createTrackbar("H Max", "HSV Ayarlari", 179, 179, nothing)
cv2.createTrackbar("S Max", "HSV Ayarlari", 255, 255, nothing)
cv2.createTrackbar("V Max", "HSV Ayarlari", 255, 255, nothing)

while True:

    # trackbar değerlerini oku
    h_min = cv2.getTrackbarPos("H Min", "HSV Ayarlari")
    s_min = cv2.getTrackbarPos("S Min", "HSV Ayarlari")
    v_min = cv2.getTrackbarPos("V Min", "HSV Ayarlari")

    h_max = cv2.getTrackbarPos("H Max", "HSV Ayarlari")
    s_max = cv2.getTrackbarPos("S Max", "HSV Ayarlari")
    v_max = cv2.getTrackbarPos("V Max", "HSV Ayarlari")

    # HSV alt ve üst sınırları oluştur
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # maskeyi ve maskelenmiş görüntüyü oluştur
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow("HSV Ayarlari", result)
    cv2.imshow("Maske", mask)

    # esc ile program çık
    if cv2.waitKey(1) & 0xFF == 27:
        break