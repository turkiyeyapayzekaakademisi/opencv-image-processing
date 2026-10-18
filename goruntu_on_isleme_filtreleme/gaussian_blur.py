import cv2

image = cv2.imread("images/gurultulu.jpg")

# (3,3): kernel boyutu, 0: sigma değeri
blur_small = cv2.GaussianBlur(image, (3, 3), 0)

# kernel büyüdükçe daha eniş komşuluk kullanılır ve yumuşatma artar
blur_medium = cv2.GaussianBlur(image, (7, 7), 0)
blur_large = cv2.GaussianBlur(image, (15, 15), 0) # görüntü daha fazla yumuşar (gürültü azalır) ama detay kaybı artars

# sonuçları karşılaştır
cv2.imshow("orijinal", image)
cv2.imshow("Gaussian 3x3", blur_small)
cv2.imshow("Gaussian 7x7", blur_medium)
cv2.imshow("Gaussian 15x15", blur_large)

# orta seviyede yumuşatılmış görüntüyü kaydet
cv2.imwrite("outputs/gaussian_blur.jpg", blur_medium)

cv2.waitKey(0)
cv2.destroyAllWindows()