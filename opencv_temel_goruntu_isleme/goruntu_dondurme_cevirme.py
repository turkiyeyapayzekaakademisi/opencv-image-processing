import cv2

image = cv2.imread("images/bird.jpg")

# görüntüyü yatay çevirme
horizontal = cv2.flip(image, 1) # 1 = görüntüyü y ekseni boyunca çevir

# görüntüyü dikey çevir
vertical = cv2.flip(image, 0) # 0 = görüntüyü x ekseni boyunca çevir

# görüntüyü hem yatay hem de dikey çevir 
both = cv2.flip(image, -1) # -1 = hem x hem y ekseni boyunca çevir

# görüntüyü saat yönünde 90 derece döndür
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("orijinal", image)
cv2.imshow("yatay", horizontal)
cv2.imshow("dikey", vertical)
cv2.imshow("iki eksende", both)
cv2.imshow("90 derece", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()