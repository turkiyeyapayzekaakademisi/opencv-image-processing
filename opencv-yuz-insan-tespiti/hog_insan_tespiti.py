import cv2

image = cv2.imread("images/insanlar.png")

if image is None:
    raise RuntimeError("goruntu okunamadi")

# çok büyük görüntüler için HOG taraması yavaş çalışır
# görüntüyü maksimum 800 px genişliğine küçültüyoruz
if image.shape[1] > 800:
    scale_ratio = 800 / image.shape[1]

    new_heigth = int(image.shape[0] * scale_ratio)

    image = cv2.resize(image, (800, new_heigth))

# Histogram of Oriented Gradients özelliklerini hesaplayabilen ve bir svm dedektörüyle nesne tespiti yapabilen bir opencv sınıfı
hog = cv2.HOGDescriptor()

# opencv içerisinde hazır gelen insan tespit svm katsayılarını verir
people_detector = cv2.HOGDescriptor_getDefaultPeopleDetector()

# HOGDescriptor nesnesine hangi svm sınıflandırıcının kullanılacağını tanımlar
hog.setSVMDetector(people_detector)

# hitThreshold: hog özelliklerinin SVM karar düzlemine göre pozitif insan tespiti kabul edilme eşiği
# padding: algılama sırasında ek çalışma alanı sağlar
# scale: her yeni ölçekte algılama penceresinin büyütülme katsayısı
# groupThreshold: aynı insan üzerinde oluşabilen benzer dikdörtgenlerin gruplanmasında kullanılan eşik
boxes, weights = hog.detectMultiScale(image, hitThreshold=0, padding=(8, 8), scale=1.05, groupThreshold=2)

# boxes: bulunan insan dikdörtgenleri
# weights: tespitlerin sınıflandırıcı skorları
for (x, y, width, heigth), weight in zip(boxes, weights):

    cv2.rectangle(image, (x,y), (x+width, y+ heigth), (0, 255,0), 2)

    confidence = float(weight)

    cv2.putText( image, f"{confidence:.2f}", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2 )

print(f"tespit edilen insan sayisi: {len(boxes)}")

cv2.imshow("hog insan tespiti", image)

cv2.imwrite("outputs/hog_insan_tespiti.jpg", image)

cv2.waitKey(0)
cv2.destroyAllWindows()



