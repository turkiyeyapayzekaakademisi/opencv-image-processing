import cv2

image = cv2.imread("images/sekiller.png")

result = image.copy()

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# gürültü giderme
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# thresholding
_, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# kontur bulma
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

shape_count = 0

for contour in contours:

    area = cv2.contourArea(contour)
    if area < 500:
        continue

    # kapalı konturun çevre uzunluğunu hesapla
    perimeter = cv2.arcLength(contour, True)

    # polygon yaklaştırma
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    # köşe sayısı belirler
    corners = len(approx)

    # nesnenin çevresindeki dikdörtgeni hesapla
    x, y, w, h = cv2.boundingRect(approx)

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

    # nesnenin momentlerini hesapla
    moments = cv2.moments(contour)

    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
    else:
        center_x = x + w // 2
        center_y = y + h // 2

    # şeklin çevresine bounding box çizdir ve merkez ve şekil adı
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(result, (center_x, center_y), 5, (0, 255, 0), -1)
    cv2.putText(result, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    shape_count += 1

# toplam tespit edilen şekil sayısı
cv2.putText(result, f"Şekil sayısı: {shape_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

cv2.imshow("Binary goruntu", binary)
cv2.imshow("otomatik sekil tespiti", result)

cv2.waitKey(0)
cv2.destroyAllWindows()