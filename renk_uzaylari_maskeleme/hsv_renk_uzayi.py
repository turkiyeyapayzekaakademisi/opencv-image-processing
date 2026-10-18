import cv2

image = cv2.imread("images/image.jpg")

# BGR görüntüyü HSV renk uzayına dönüştür
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# görüntünün merkez piksellerinin incelenmesi
height, width = image.shape[:2]
x, y = width // 2, height // 2

b, g, r = image[y, x]
h, s, v = hsv[y, x] # h: rengin kendisi, s: rengin doygunluğu, v: rengin parlaklığı

print(f"BGR: b: {b}, g: {g}, r: {r}")
print(f"HSV: h: {h}, s: {s}, v: {v}")

# HSV kanallarını ayrı ayrı çıkar
hue, saturation, value = cv2.split(hsv)

# sonuçları görselleştir
cv2.imshow("orijinal bgr", image)
cv2.imshow("hue: renk", hue)
cv2.imshow("saturation: doygunluk", saturation)
cv2.imshow("value: parlaklik", value)

cv2.waitKey(0)
cv2.destroyAllWindows()

