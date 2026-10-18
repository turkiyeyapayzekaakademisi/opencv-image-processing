import cv2

image = cv2.imread("images/sekiller.png")

result = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# threshold
# THRESH_BINARY_INV: koyu nesneleri açık, beyaz arka planı ise siyah yapar
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Kontur bulma
# RETR_EXTERNAL: dış konturleri (dış sınırları) bul
# CHAIN_APPROX_SIMPLE: gereksiz ara noktaları saklamayarak konturu sadeleştirir
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Bulunan kontur sayısı: {len(contours)}")

# -1: bulunan tüm konturları çizdir
# 2: çizgi kalınlığı
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

cv2.imshow("binary", binary)
cv2.imshow("bulunan konturlar", result)

cv2.imwrite("outputs/konturlar.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()