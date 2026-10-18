import cv2

image = cv2.imread("images/detay.png", cv2.IMREAD_GRAYSCALE)

# nfeatures: anahtar sayısı
# scaleFactor: görüntü pramidinde ardışık seviyeler arasındaki ölçek oranı
# nlevels: ORB'nin inceleyeceği görüntü piramidi seviye sayısı
orb = cv2.ORB_create(nfeatures = 100, scaleFactor = 1.2, nlevels = 8)


# detectAndCompute: anahtar nokta tespit ve descriptor hesaplama
# None: anahtar nokta aranacak özel bir maske kullanılmıyor
keypoints, descriptors = orb.detectAndCompute(image, None)

# anahtar noktaları görüntü üzerinde göster
result = cv2.drawKeypoints(image, keypoints, None, color = (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

if descriptors is not None:
    print(f"descriptor boyutu: {descriptors.shape}")
    print(f"ilk descriptor: \n{descriptors[0]}")

cv2.imshow("orb anahtar noktalari", result)
cv2.imwrite("outputs/orb_anahtar_noktalari.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
