import cv2

image = cv2.imread("images/bird.jpg")

height, width = image.shape[:2]

# ROI sınırları belirle
x1, x2 = width // 4, 3 * width // 4
y1, y2 = height // 4, 3 * height // 4

# görüntünün yalnızca seçilen bölgesini al
roi = image[y1:y2, x1:x2]

# ROI içerisinde ortalama BGR değerlerini hesapla
mean_blue, mean_green, mean_red, _ = cv2.mean(roi)

print(f"mean_blue: {mean_blue}")
print(f"mean_green: {mean_green}")
print(f"mean_red: {mean_red}")

# ROI'nin görüntünün neresinde olduğunu göstermek için alan çiz
result = image.copy()
cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2) # x1, y1 den x2, y2 ye kadar yeşil ve kalınlığı 2 piksel olan bir dikdörtgen

cv2.imshow("ROI Bolgesi", result)
cv2.imshow("sadece ROI", roi)

cv2.waitKey(0)
cv2.destroyAllWindows()