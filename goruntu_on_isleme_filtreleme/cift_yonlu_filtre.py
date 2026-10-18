import cv2

image = cv2.imread("images/iki_tarafli.jpg")


# d: her piksel için değerlendirilecek komşuluk çapı
# sigmaColor: renkleri ne kadar farklı piksellerin birlikte filtrelenebileceği
# sigmaSpace: fiziksel olarak ne kadar uzaktaki piksellerin dikkate alınacağı
cift_yonlu = cv2.bilateralFilter(image, d = 9, sigmaColor=75, sigmaSpace= 75)

# karşılaştırmak için gaussian blur 
gaussian = cv2.GaussianBlur(image, (9,9), 0)

# sonuçları göster
cv2.imshow("Orijinal", image)
cv2.imshow("gaussian blur", gaussian)
cv2.imshow("cift yonlu", cift_yonlu)

# çift yönlüyü kaydet
cv2.imwrite("outputs/cift_yonlu.jpg", cift_yonlu)
cv2.waitKey(0)
cv2.destroyAllWindows()
