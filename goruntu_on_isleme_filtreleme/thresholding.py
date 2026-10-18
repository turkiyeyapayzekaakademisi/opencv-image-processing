import cv2

gray = cv2.imread("images/belge.jpg", cv2.IMREAD_GRAYSCALE)

# 127: eşik değeri (threshold) 
# 255: eşiği geçen piksellere atanacak maksimum değer 
# THRESH_BINARY: eğer piksel value > 127 ise 255 (beyaz), küçükse 0 (siyah) 
threshold_value, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 

# THRESH_BINARY_INV: binary işlemin tersi
_, binary_inverse = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Sonuçları görselleştir
cv2.imshow("gri seviye", gray)
cv2.imshow("THRESH_BINARY", binary)
cv2.imshow("THRESH_BINARY_INV", binary_inverse)

# binary görüntüyü kaydet
cv2.imwrite("outputs/binary.jpg", binary)

cv2.waitKey(0)
cv2.destroyAllWindows()