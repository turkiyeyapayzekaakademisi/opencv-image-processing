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

print(f"goruntu boyutu: {image.shape}")

# scalefactor: piksel değerlerini modelin eğitim sırasında kullandığı ölçeğe dönüştürür. 1 / 127.5 değerinde
# size: mobilenet ssd modelinin beklediği giriş boyutu
# mean: piksel değerlerinden çıkarılacak ortamala değerler
# swapRB: BGR to RGB YAPMA 
# crop: görüntünün merkezden kırpılmasını istemiyoruz
blob = cv2.dnn.blobFromImage(image, scalefactor=0.007843, size=(300, 300), mean = (127.5, 127.5, 127.5), swapRB=False, crop=False)

print(f"blob boyutu: {blob.shape}")

cv2.imshow("orijinal goruntu", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""
(H, W, C) -> (N, C, H, W)
"""