import cv2
import numpy as np

image = cv2.imread("images/cizgiler.png")

result = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# küçük gürültüleri azaltmak için gaussian kernel oluştur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# hough transform
# rho: uzaklık çözünürlüğü
# theta: açı çözünürlüğü
# threshold: bir yapının çizgi olarak kabul görmesi için gereken min oy
# minLineLength: kabul edilecek min çizgi uzunluğu
# maxLineGap: iki çizgi parçasının birleşebilmesi için izin verilen maksimum boşluk
lines = cv2.HoughLinesP(edges, rho = 1, theta = np.pi / 180, threshold=60, minLineLength=50, maxLineGap=10)

if lines is not None:
    for line in lines:

        x1, y1, x2, y2 = line.ravel()
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

print(f"Bulunan çizgi sayısı: {len(lines)}")

cv2.imshow("Canny kenarlari", edges)
cv2.imshow("Hough cizgileri", result)

cv2.imwrite("outputs/hough_cizgileri.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()