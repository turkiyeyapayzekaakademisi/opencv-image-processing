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

# nesnelerin bulunduğu video dosyasını aç
capture = cv2.VideoCapture("videos/nesneler.mp4")

if not capture.isOpened():
    raise RuntimeError("video acilmadi")

# confidence threshold: modelin yeterince güvenmediği tahminleri elemek için bir eşik değeri
confidence_threshold = 0.55

while True:

    success, frame = capture.read()

    if not success:
        break

    # Model sonuçlarını gerçek piksel koordşnatlarında çevirmek için gerekli
    height, width = frame.shape[:2]

    # her video karesini mobileNet SSD modelinin beklediği giriş formatına çevir
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.007843, size=(300, 300), 
                          mean = (127.5, 127.5, 127.5), swapRB=False, crop=False)

    # o anki frame'e ait blob'u model girişine ver
    net.setInput(blob)

    # sinir ağı üzerinden inference işlemi gerçekleştir
    detections = net.forward()

    # modelin ürettiği bütün detection sonuçlarını incele
    for i in range(detections.shape[2]):

        # tahmine ait confidence score al
        confidence = float(detections[0, 0, i, 2])

        if confidence < confidence_threshold:
            continue

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
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # nesne sınıfı ve confidence bilgisini etiket olarak oluştur
        label = (
            f"{class_name}: "
            f"{confidence:.2f}"
        )

        cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # dnn ile işlenmiş video karesini göster
    cv2.imshow("dnn video nesne tespiti", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
