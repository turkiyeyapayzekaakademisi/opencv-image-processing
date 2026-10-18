import cv2
import numpy as np

def order_points(points):

    # çıktı sonrası için sol üst, sağ üst, sağ alt, sol alt
    ordered = np.zeros((4,2), dtype = np.float32)

    # x + y toplamı en küçük nokta sol üst ve en büyük nokta sağ alt olacak şekilde 
    point_sum = points.sum(axis=1)
    ordered[0] = points[np.argmin(point_sum)] # sol üst
    ordered[2] = points[np.argmax(point_sum)] # sağ alt

    # y - x farkı en küçük nokta sağ üst ve en büyük nokta sol alt olur
    point_diff = np.diff(points, axis = 1).reshape(-1)
    ordered[1] = points[np.argmin(point_diff)] # sağ üst
    ordered[3] = points[np.argmax(point_diff)] # sol alt

    return ordered

def four_point_transform(image, points):

    # 4 noktayı standart sıraya getir
    top_left, top_right, bottom_right, bottom_left = order_points(points)

    # alt ve üst kenarların uzunluğunu hesapla
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    width_top = np.linalg.norm(top_right - top_left)
    max_width = int(max(width_bottom, width_top))

    # sol ve sağ kenarların uzunluğunu hesapla
    height_right = np.linalg.norm(top_right - bottom_right)
    height_left = np.linalg.norm(top_left - bottom_left)
    max_height = int(max(height_right, height_left))

    # düzeltilmiş görüntünün 4 köşesini oluştur
    destination = np.float32(
        [
            [0, 0], 
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ]
    )

    # kaynak ve hedef 4 noktadan perspektif matrisi oluştur
    matrix = cv2.getPerspectiveTransform(np.float32([top_left, top_right, bottom_right, bottom_left]), destination)

    return cv2.warpPerspective(image, matrix, (max_width, max_height))

image = cv2.imread("images/belge.png")
height, width = image.shape[:2]

# belgenin 4 köşesini ell belirle
document_points = np.float32(
    [
        [160, 160], # sol üst
        [550, 92], # sağ üst
        [870, 530], # sağ alt
        [304, 673], # sol alt

    ]
)

# belgenin perspective ini düzelt
warped = four_point_transform(image, document_points)

marked = image.copy()

for x, y in document_points.astype(int):
    cv2.circle(marked, (x,y), 8, (0,0,255), -1)

cv2.imshow("belge ve dort nokta", marked)
cv2.imshow("duzeltilmis belge", warped)

cv2.imwrite("outputs/dort_nokta_belge.jpg", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()