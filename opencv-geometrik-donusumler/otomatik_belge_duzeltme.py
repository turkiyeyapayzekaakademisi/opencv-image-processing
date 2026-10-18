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

def perspective_transform(image, points):

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
result = image.copy()

# görüntüyü gri seviyeye dönüştür
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# gürültü azaltma
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Beyaz yani açık renkli kağıdı masadan tamamen ayır
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# kenarları birleştirmek için morphological closing
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# kontur bulma
contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# konturları büyükten küçüğe sırala
contours = sorted(contours, key = cv2.contourArea, reverse=True)[:10]

document_contour = None
for contour in contours:

    # çevre hesaplaması yap
    perimeter = cv2.arcLength(contour, True)

    # poligon yakınlaştırması
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 4:
        document_contour = approx
        break

if document_contour is None:
    raise RuntimeError("Belge sınırı bulunamadı")

# belgenin 4 köşesini (4, 2) matris formatına çevir
docucment_points = document_contour.reshape(4, 2).astype(np.float32)

# belgenin perspective düzelt
scanned = perspective_transform(image, docucment_points)

# kontur çizdir
cv2.drawContours(result, [document_contour], 0, (0, 255, 0), 3)

# bulunan köşe noktalarını ayrıca göster
for point in document_contour:
    x, y = point[0]
    cv2.circle(result, (x, y), 8, (0, 0, 255), -1)

cv2.imshow("Canny kenarlari", closed)
cv2.imshow("Tespit edilen belge", result)
cv2.imshow("perspective duzeltilmis belge", scanned)

cv2.imwrite("outputs/belge_tespiti.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
