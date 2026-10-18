import cv2 
import numpy as np

image = cv2.imread("images/daireler.png")
result = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# median blur ile gürültü azaltma
gray = cv2.medianBlur(gray, 5)

# HOUGH_GRADIENT: opencv'nin gradient tabanlı hough circle yöntemi
# dp: accumulator çözünürlüğünün görüntü çözünürlüğüne oranı
# minDist: tespit edilen iki daire merkezi arasında ki minimum mesafe
# param1: içeride kullanılan Canny'nin yüksek eşik değeri
# param2: daire kabulu için accumulatır eşiği, düşerse daha fazla daire buluruz
# minRadius ve maxRadius: kabul edilecek min ve max yarı çap
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp = 1.2, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=150)

if circles is not None:

    circles = np.uint16(np.around(circles)) # koordinat ve yarı çapları tam sayıya dönüştür

    for x, y, radius in circles[0]:
        # dairenin dış sınırlarını yeşil çiz
        cv2.circle(result, (x,y), radius, (0, 255, 0), 2)

        # dairenin merkez noktasını kırmızı gösterelim
        cv2.circle(result, (x, y), 4, (0, 0, 255), -1)

    print(f"Bulunan daire sayısı: {len(circles[0])}")

    cv2.imshow("hough circle", result)
    cv2.imwrite("outputs/hough_daireleri.jpg", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()