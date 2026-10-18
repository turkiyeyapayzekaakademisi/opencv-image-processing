import cv2
import numpy as np

# görüntü oku
image = cv2.imread("images/perspektif.png")

height, width = image.shape[:2]

# kaynak görüntünün 4 köşesini belirle
# sıralama: sol üst, sağ üst, sağ alt ve sol alt
source_points = np.float32(
    [
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ]
)

# dört köşenin dönüşüm sonrasında gideceği noktalar
# köşeleri kaydırarak görüntüye perspektif etkisi veriyoruz
destination_points = np.float32(
    [
        [width * 0.15, height * 0.1],
        [width *0.85, 0],
        [width - 1, height - 1],
        [0, height *0.95]
    ]
)

# 4 tane kaynak ve 4 tane hedef noktadan 3x3 lük bir perspektif dönüşüm matrisi oluştur
matrix = cv2.getPerspectiveTransform(source_points, destination_points)

# perspective transform
perspective = cv2.warpPerspective(image, matrix, (width, height))

cv2.imshow("orijinal", image)
cv2.imshow("perspective transformation", perspective)

cv2.imwrite("outputs/perspective_transform.jpg", perspective)

cv2.waitKey(0)
cv2.destroyAllWindows()