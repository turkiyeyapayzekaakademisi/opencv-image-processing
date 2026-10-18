import cv2 

# görüntüyü oku
image = cv2.imread("images/belge.png")

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

# kontur çizdir
cv2.drawContours(result, [document_contour], 0, (0, 255, 0), 3)

# bulunan köşe noktalarını ayrıca göster
for point in document_contour:
    x, y = point[0]
    cv2.circle(result, (x, y), 8, (0, 0, 255), -1)

cv2.imshow("Canny kenarlari", closed)
cv2.imshow("Tespit edilen belge", result)

cv2.imwrite("outputs/belge_tespiti.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()


