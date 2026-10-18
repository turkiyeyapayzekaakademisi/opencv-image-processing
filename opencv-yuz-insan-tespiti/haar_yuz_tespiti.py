import cv2

# görüntüyü oku
image = cv2.imread("images/yuz.png")

if image is None:
    raise RuntimeError("goruntu okunamadi")

# haarcascade
face_cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

# cascade classifier
# CascadeClassifier: daha önceden eğitilmiş cascade sınıflandırıcı
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise RuntimeError("yuz cascade modeli yuklenemedi")

# Haar cascade tespitleri genellikle gri seviye görüntü üzerinde yapılır.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# histogram eşitleme
# haar özelliklerinin daha belirgin hale gelmesine yardımcı olur
gray = cv2.equalizeHist(gray)

# scaleFactor: her tarama seviyesinde görüntünün ne kadar küçüleceğini belirtir
# minNeighbors: bir bölgenin yüz kabul edilebilmesi için çevresinde bulunması gereken minimum komşu tespit sayısını belirler
# minSize: bundan küçük bölgelerin yüz olarak değerlendirimemesini sağlar
faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30))

for x, y, width, height in faces:

    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

print(f"tespit edilen yüz sayısı: {len(faces)}")

cv2.imshow("Yüz tespiti", image)

cv2.imwrite("outputs/yuz_tespiti.jpg", image)

cv2.waitKey(0)
cv2.destroyAllWindows()