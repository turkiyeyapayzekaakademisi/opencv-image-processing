import cv2
import numpy as np

# görüntü oku
image = cv2.imread("images/manzara.png")

height, width = image.shape[:2]

# kaynak görüntüden birbirine karşılık gelmeyen 3 nokta seç
# sırasıyla: sol üst, sağ üst, sol alt köşeler
source_points = np.float32(
    [
        [0, 0], 
        [width - 1, 0],
        [0, height - 1]
    ]
)

# kaynak noktaların dönüşüm sonrasında gideceği yeni konumlar
destination_points = np.float32(
    [
        [width * 0.15, height * 0.1],
        [width * 0.85, height * 0.2],
        [width * 0.1, height * 0.9],
    ]
)

# 3 kaynak ve 3 hedef noktadan 2x3 lük bir affine dönüşüm matrisi oluştur
matrix = cv2.getAffineTransform(source_points, destination_points)

# matrix kullanılacak dönüşüm matrisi
# (width, height): oluşturulacak çıktının boyutu
affine = cv2.warpAffine(image, matrix, (width, height))

cv2.imshow("orijinal", image)
cv2.imshow("affine", affine)

cv2.imwrite("outputs/affine_transform.jpg", affine)

cv2.waitKey(0)
cv2.destroyAllWindows()