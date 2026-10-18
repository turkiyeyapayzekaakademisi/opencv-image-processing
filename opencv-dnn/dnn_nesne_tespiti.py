import cv2

# MobileNet SSD modelinin eğitimde kullanılan sınıfları var
# listenin sırasını modelin döndürdüğü class_id değerleriyle eşleme yapalım
classes = [ 
    "background", 
    "aeroplane", 
    "bicycle", 
    "bird", 
    "boat", 
    "bottle", 
    "bus", 
    "car", 
    "cat", 
    "chair", 
    "cow", 
    "diningtable", 
    "dog", 
    "horse", 
    "motorbike", 
    "person", 
    "pottedplant", 
    "sheep", 
    "sofa", 
    "train", 
    "tvmonitor" ]

# model dosyalarının yollarını belirle
config_path = "models/MobileNetSSD_deploy.prototxt"
model_path = "models/MobileNetSSD_deploy.caffemodel"

# mobileNet SSD yükle
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

if net.empty():
    raise RuntimeError("dnn modeli yuklenmedi")

# test gürüntüsünü oku
image = cv2.imread("images/nesneler.png")

if image is None:
    raise RuntimeError("goruntu okunmadi")

height, width = image.shape[:2]

# model girişi için blob oluştur
blob = cv2.dnn.blobFromImage(image, scalefactor=0.007842, size=(300,300), mean=(127.5, 127.5, 127.5),swapRB=False, crop=False)

# hazırlanan blob u modele girdi olarak ver
net.setInput(blob)

# ağ üzerinde forward pass gerçekleştir ve tahminleri al
detections = net.forward()

# confidence threshold: modelin yeterince güvenmediği tahminleri elemek için bir eşik değeri
confidence_threshold = 0.55

for i in range(detections.shape[2]):

    # detection içerisindeki 3. değer bizim confidence
    confidence = float(detections[0, 0, i, 2])

    if confidence < confidence_threshold:
        continue

    # detection içerisindeki 2. değer sınıf id
    class_id = int(detections[0, 0, i, 1])

    if class_id < 0 or class_id >= len(classes):
        continue

    # detection içerisindeki koordinatlar 0-1 arasında gelir
    # bunları görüntünün width ve height değerleriyle çarparak gerçek piksel koordinatlarına dönüştür
    x1 = int(detections[0, 0, i, 3] * width)
    y1 = int(detections[0, 0, i, 4] * height)
    x2 = int(detections[0, 0, i, 5] * width)
    y2 = int(detections[0, 0, i, 6] * height)

    # koordinatların görüntü sınırlarının dışına taşmasını önle
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    # class_id kullanarak nesnenin sınıf ismini bul
    class_name = classes[class_id]

    # tespit edilen nesnenin çevresine bounding box çiz
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # nesne sınıfı ve confidence bilgisini etiket olarak oluştur
    label = (
        f"{class_name}: "
        f"{confidence:.2f}"
    )

    cv2.putText(image, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("dnn nesne tespiti", image)

cv2.imwrite("outputs/dnn_nesne_tespiti.jpg", image)

cv2.waitKey(0)
cv2.destroyAllWindows()