import cv2

image = cv2.imread("images/bird.jpg")

# orijinal görüntüyü değiştirmemek için kopyasını oluştur
result = image.copy()
height, width = result.shape[:2]

# görüntünün üst kısmına mavi çizgi çiz
cv2.line(result, (20, 20), (width - 20, 20), (255, 0, 0), 3)

# görüntünün sol tarafına yeşil dikdörtgen çiz
cv2.rectangle(result, (50, 50), (width // 2, height // 2), (0, 255, 0), 3) 

# görüntünün sağ tarafına kırmızı daire çiz
center = (3 * width // 4, height // 2) # dairenini merkezi
radius = min(width, height) // 8
cv2.circle(result, center, radius, (0, 0, 255), 3)

# görüntünün merkezine dolu bir nokta ekle
cv2.circle(result, (width // 2, height // 2), 6, (0, 255, 255), -1) # -1 içi dolu

cv2.imshow("Geometrik şekiller", result)
cv2.waitKey(0)
cv2.destroyAllWindows()