import cv2

# modelin ağ mimarisini tanımlayan prototxt dosyası
config_path = "models/MobileNetSSD_deploy.prototxt"

# modelin eğitim sonucunda öğrenilmiş ağırlıklarını içeren dosya
model_path = "models/MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(config_path, model_path)
if net.empty():
    raise RuntimeError("DNN modeli yuklenemedi")

# görüntü okutma
image = cv2.imread("images/nesneler.png")

if image is None:
    raise RuntimeError("goruntu okunamadi")

# görüntüyü modelin beklediği blob yapısına dönüştür
blob = cv2.dnn.blobFromImage(image, scalefactor=0.007843, size=(300, 300), mean = (127.5, 127.5, 127.5), swapRB=False, crop=False)

# oluşturduğumuz blobu sinir ağının girişine yerleştir
net.setInput(blob)

# görüntüyü sinir ağının bütün katmanlarından geçirir ve modelin tahmin sonuçlarını döndürür
detections = net.forward()

print(f"tahmin tensor boyutu: {detections.shape}")

print(f"ilk tahmin: {detections[0, 0, 0]}")

"""
ilk tahmin: 
[ 0.         15.          0.9999424   0.2443057   0.0267657   0.44190267    0.785033  ]
batch_id    class_id    confidence    x1            y1          x2              y2
"""