import cv2

image = cv2.imread("images/sekiller.png")

result = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:

    # küçük konturları filtrele
    area = cv2.contourArea(contour)

    if area < 500: 
        continue

    x, y, w, h = cv2.boundingRect(contour)

    # (x, y): başlangıç noktası kutunun köşesi
    # (x + w, y + h): bitiş noktası
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0))

    # kutunun merkezini hesapla
    center = (x + w // 2, y + h // 2)
    cv2.circle(result, center, 5, (0, 0, 255), -1)

cv2.imshow("bounding rectangle", result)
cv2.imwrite("outputs/bounding_rectangle.png", result)

cv2.waitKey(0)
cv2.destroyAllWindows()