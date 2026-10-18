import cv2

reference = cv2.imread("images/referans.png", cv2.IMREAD_GRAYSCALE)
scene = cv2.imread("images/sahne.png", cv2.IMREAD_GRAYSCALE)

# orb
orb = cv2.ORB_create(nfeatures = 100)

keypoints_reference, descriptors_reference = orb.detectAndCompute(reference, None)
keypoints_scene, descriptors_scene = orb.detectAndCompute(scene, None)

if descriptors_reference is None or descriptors_scene is None:
    raise RuntimeError("yeterli ozellik bulunamadi")

# Brute force eşleştirme
# NORM_HAMMING: orb nin binary descriptorları arasındaki farkı ölçer
# crossCheck: A ile B eşleşmesinin B ile A yönünde en iyi eşleşme olmasını ister 
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# her descriptor için karşı görüntüde ki en iyi eşleşmeyi bul
matches = matcher.match(descriptors_reference, descriptors_scene)

matches = sorted(matches, key = lambda match: match.distance)

# görüntünün okunabilir kalması için en iyi 20 eşleşme
best_matches = matches[:20]

result = cv2.drawMatches(reference, keypoints_reference, scene, keypoints_scene, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("brute force eslesmeleri", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

