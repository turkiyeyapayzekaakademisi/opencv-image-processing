import cv2
import numpy as np

# referans nesneyi ve nesnenin aranacağı sahneyi oku
reference = cv2.imread("images/referans1.png", cv2.IMREAD_GRAYSCALE)
scene = cv2.imread("images/sahne1.png", cv2.IMREAD_GRAYSCALE)

# sonucu renkle göstermek için sahnenin bgr kopyasının oluşturulması
scene_result = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)

# orb nesnesi
orb = cv2.ORB_create(nfeatures = 2000)

# referans ve sahne görüntülerinden anahtar noktaları ve descriptorların bulunması
keypoints_reference, descriptor_reference = orb.detectAndCompute(reference, None)
keypoints_scene, descriptor_scene = orb.detectAndCompute(scene, None)

if descriptor_reference is None or descriptor_scene is None:
    raise RuntimeError("yeterli ozellik bulunamadi")

# Brute force matcher
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# knn matcher ile en yakın adayları bul
knn_matches = matcher.knnMatch(descriptor_reference, descriptor_scene, k = 2)

good_matches = []

for match_pair in knn_matches:

    if len(match_pair) < 2:
        continue

    first, second = match_pair

    if first.distance < 0.75 * second.distance:
        good_matches.append(first)

# homography: hesaplamak için en az 4 nokta eşleşmesine ihtiyaç var
if len(good_matches) < 4:
    raise RuntimeError("nesneyi bulmak için yeterli eşleşme yok")

# iyi eşleşmelerden koordinatları çekme adımı
# Referans görüntüden noktasal koordinatları çekme
# queryIdx: eşleşmenin referans görseldeki keypoint indexi
# pt: x ve y piksel koordinatları
# [N, 1, 2]: N -> nokta sayısı, 1 -> kanal, 2 -> x ve y
source_points = np.float32(
    [keypoints_reference[match.queryIdx].pt for match in good_matches]
).reshape(-1, 1, 2)

# sahne görüntüsündeki noktasal koordinatlar
# tainIdx: eşleşmenin sahne görselindeki keypoint indexini verir
destination_points = np.float32([keypoints_scene[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

# homography
# ransac: yanlış eşleşme etkisi azaltma
# 5: bir eşleşmenin doğru kabulü için maksimum reprojection hata eşiği
homography, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5)

if homography is None:
    raise RuntimeError("homography hesaplanamadi")

# referans görüntünün yüksekliğini ve genişliğini al
height, width = reference.shape

# referans görüntünün 4 köşesini oluştur
reference_corners = np.float32(
    [
        [0,0],
        [width -1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ]
).reshape(-1, 1, 2)

# homography ile referans köşelerin sahnede ki karşılıklarını hesapla
scene_corners = cv2.perspectiveTransform(reference_corners, homography)

# köşeleri çizim için tam sayı koordinatlarına dönüştür
scene_corners = np.int32(scene_corners)

# True: 4 noktayı kapalı poligon olarak birleştirir
cv2.polylines(scene_result, [scene_corners], True, (0, 255, 0), 4)

# RANSAC maskesini python listesine dönüştür
inlier_mask = mask.ravel().tolist()

# eşleşmeleri ve bulunan nesne bölgesini birlikte göster
match_result = cv2.drawMatches(
    reference,
    keypoints_reference,
    scene_result, 
    keypoints_scene,
    good_matches,
    None,
    matchColor=(0,255,0),
    singlePointColor=None,
    matchesMask=inlier_mask,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

print(f"iyi eslesme sayisi: {len(good_matches)}")
print(f"RANSAC sonrasi kullanılan eşleşme: {sum(inlier_mask)}")

cv2.imshow("bulunan nesne", scene_result)
cv2.imshow("eslesmeler ve nesne konumu", match_result)

cv2.imwrite("outputs/bulunan_nesne.jpg", scene_result)
cv2.imwrite("outputs/nesne_eslesmeleri.jpg", match_result)

cv2.waitKey(0)
cv2.destroyAllWindows()