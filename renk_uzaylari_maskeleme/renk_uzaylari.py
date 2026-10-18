import cv2

# görüntüyü BGR formatında oku 
image = cv2.imread("images/image.jpg")

# BGR görüntüyü RGB ve Gray Scale'e çevir
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(f"BGR boyutu: {image.shape}")
print(f"RGB boyutu: {rgb.shape}")
print(f"gray boyutu: {gray.shape}") # (1054, 1408, 1)

# merkez pikselin değerini karşılaştır
height, width = image.shape[:2]
x, y = width // 2, height // 2
print(f"BGR piksel: {image[y,x]}") # [  0  87 225]
print(f"RGB piksel: {rgb[y,x]}")   # [225  87   0]
print(f"gray piksel: {gray[y,x]}") # 118

# görsel oluştur
cv2.imshow("RGB Goruntu", image)
cv2.waitKey(0)
cv2.destroyAllWindows()