import cv2

# görüntüyü oku
image = cv2.imread("images/sekiller.png")

result = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV) 

# dış konturları bul
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:

    area = cv2.contourArea(contour)

    if area < 500:
        continue

    # kapalı konturun çevre uzunluğunu hesapla
    perimeter = cv2.arcLength(contour, True)

    # 0.02 * perimeter: yaklaştırma hassasiyeti
    # değer küçüldükçe daha fazla, büyüdükçe daha az köşe korunuyor
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    corners = len(approx)

    if corners == 3:
        shape = "ucgen"
    elif corners == 4:
        # dörtgenin genişlik ve yükseklik oranını kullanarak kare-dikdörtgen ayrım yapalım
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        # oran 1 e yakınsa kare kabul et
        shape = "kare" if 0.95 <= aspect_ratio <= 1.05 else "dikdortgen"
    elif corners == 5:
        shape = "besgen"
    else:
        shape = "daire"

    # seklin merkezini belirle
    x, y, w, h = cv2.boundingRect(approx)

    # yaklaştırılmış kontur çiz
    cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)

    # şekil adını görüntü üzerine ekle
    cv2.putText(result, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imshow("geometrik şekil tespiti", result)
cv2.imwrite("outputs/geometrik_sekiller.png", result)

cv2.waitKey(0)
cv2.destroyAllWindows()