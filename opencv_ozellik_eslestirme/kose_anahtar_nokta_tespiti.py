import cv2
import numpy as np

image = cv2.imread("images/detay.png")

# iki farklı sonucu ayrı ayrı gösterebilmek için kopyalar oluştur
corner_result = image.copy()
keypoint_result = image.copy()

# gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# köşe hesabı
# maxCorners: bulunabilecek maksimum köşe sayısı
# qualityLevel: en güçlü köşeye göre kabul edilecek min kalite oranı
# minDistance: bulunan iki köşe arası min piksel mesafesi
# blockSize: köşe hesabında kullanılacak komşuluk alanı 
corners = cv2.goodFeaturesToTrack(gray, maxCorners=30, qualityLevel=0.1, minDistance=40, blockSize=3)

if corners is not None:
    corners = np.int32(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(corner_result, (x, y), 5, (0, 0, 255), -1)

# feature detector
# threshold: FAST için merkez ve çevre pikseller arasında ki min yoğunluk
# nonmaxSupression: birbirine çok yakın zayıf noktaları bastır
fast = cv2.FastFeatureDetector_create(threshold = 95, nonmaxSuppression = True)

# fast ile anahtar noktaları tespit et
keypoints = fast.detect(gray, None)

# DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: anahtar noktaların konum ve ölçek bilgilerini daha belirgin yap
keypoint_result = cv2.drawKeypoints(image, keypoints, None, color = (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("orijinal", image)
cv2.imshow("kose noktalari", corner_result)
cv2.imshow("fast anahtar noktalari", keypoint_result)

cv2.imwrite("outputs/kose_noktalari.jpg", corner_result)
cv2.imwrite("outputs/anahtar_noktalari.jpg", keypoint_result)

cv2.waitKey(0)
cv2.destroyAllWindows()