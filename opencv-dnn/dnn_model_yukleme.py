import cv2

# https://github.com/Qengineering/MobileNetV1_SSD_OpenCV_Caffe?utm_source=chatgpt.com

# modelin ağ mimarisini tanımlayan prototxt dosyası
config_path = "models/MobileNetSSD_deploy.prototxt"

# modelin eğitim sonucunda öğrenilmiş ağırlıklarını içeren dosya
model_path = "models/MobileNetSSD_deploy.caffemodel"

# caffe formatındaki hazır bir derin öğrenme modelini opencv dnn modülüne yükler
# fonksiyon bize cv2.dnn_net tipinde bir ağ nesnesi döndürür
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

if net.empty():
    raise RuntimeError("DNN modeli yuklenemedi")

print(type(net))