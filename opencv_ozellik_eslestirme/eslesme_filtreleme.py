import cv2

# görüntüleri gri seviyede oku
reference = cv2.imread("images/referans.png", cv2.IMREAD_GRAYSCALE)
scene = cv2.imread("images/sahne.png", cv2.IMREAD_GRAYSCALE)

# nfeatures: farklı ölçek ve açıkarda yeterli özellik bulabilmek için üst sınır
orb = cv2.ORB_create(nfeatures = 2000)

# anahtar noktaları ve orb descriptorlarını çıkar
keypoints_reference, descriptors_reference = orb.detectAndCompute(reference, None)
keypoints_scece, descriptors_scene = orb.detectAndCompute(scene, None)

if descriptors_reference is None or descriptors_scene is None:
    raise RuntimeError("yeterli ozellik bulunamadı")

# Brute force matcher
# NORM_HAMMING: orb binary descriptorlar için hamming mesafesi kullanmak
# crossCheck: knnMatch ile birden fazla aday karşılaştıracağımız için false yaptık
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# k = 2: her descriptor için en yakın 2 eşleşmeyi getir
knn_matches = matcher.knnMatch(descriptors_reference, descriptors_scene, k = 2)

good_matches = []

for match_pair in knn_matches:

    if len(match_pair) < 2: # bazı descriptorlar için 2 aday bulunmayabilir
        continue

    first, second = match_pair

    # 0.75: en iyi eşleşme ikinci adaydan belirgin şekilde daha iyi olmalı
    # oran küçüldükçe filtre daha katı hale gelir
    if first.distance < 0.75 * second.distance:
        good_matches.append(first)

# iyi eşleşmeleri mesafelerine göre sırala
good_matches = sorted(good_matches, key = lambda match: match.distance)

# en fazla 50 tane iyi eşleşme görüntüle
matches_to_draw = good_matches[:50]

# eşleşmeyen noktaları göstermeden bağlantı çizgilerini oluştur
result = cv2.drawMatches(reference, keypoints_reference, scene, keypoints_scece, matches_to_draw, None, matchColor=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

print(f"toplam knn eşleme grubu: {len(knn_matches)}")
print(f"filtreleme sonrası iyi eşleşmeler: {len(good_matches)}")

cv2.imshow("filtrelenmis eslesmeler", result)
cv2.imwrite("outputs/filtrelenmis_eslesmeler.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
