import cv2

gray = cv2.imread("images/belge.jpg", cv2.IMREAD_GRAYSCALE)

# küçük gürültüleri azalt
gray = cv2.medianBlur(gray, 5)

# karşılaştırma için thresholding
_, global_threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# adaptive threshold
# ADAPTIVE_THRESH_MEAN_C: komşuluk piksellerin ortalamasından C çıkarılacak eşik hesaplanır
# 11: blockSize, piksel çevresinde kullanılacak alan
# 2: C değeri, hesaplanan yerel eşikten çıkarılacak olan değer
adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# adaptive threshold
adaptive_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# sonuçları göster
cv2.imshow("gri seviye", gray)
cv2.imshow("global_threshold", global_threshold)
cv2.imshow("adaptive_mean", adaptive_mean)
cv2.imshow("adaptive_gaussian", adaptive_gaussian)

# gaussian adaptive threshold kaydet
cv2.imwrite("outputs/adaptive_threshold.jpg", adaptive_gaussian)

cv2.waitKey(0)
cv2.destroyAllWindows()