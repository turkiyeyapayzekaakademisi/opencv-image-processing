import cv2

image = cv2.imread("images/sekiller.png")

result = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:

    # konturun kapladığı piksel alanı
    area = cv2.contourArea(contour)

    # kontur çevre uzunluğu
    perimeter = cv2.arcLength(contour, True) # kontur kapalı bir şekilde

    if area < 500: # çok küçük alanları gürültü olarak kabul et
        continue

    # konturun görüntü momentlerini hesapla: şekil, boyut, konum ve yön
    moments = cv2.moments(contour)

    # m00 (toplam alan) sıfır değilse merkez koordinatlarını hesapla
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"]) # kontur içindeki tüm piksellerin x koordinatlarının toplamı
        center_y = int(moments["m01"] / moments["m00"]) 

        # merkez noktasını kırmızı olarak göster
        cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)

        # alan bilgisini nesnenin yanına yaz
        cv2.putText(result, f"Alan: {int(area)}", (center_x + 10, center_y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0,255, 0), 1)

# konturları çizdir
cv2.drawContours(result, contours, -1, (255, 0, 0), 2)

cv2.imshow("Kontur ozellikleri", result)

cv2.waitKey(0)
cv2.destroyAllWindows()