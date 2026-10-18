import cv2
import numpy as np

# Video karelerini en-boy oranını koruyarak küçültür
def resize_frame(frame, target_width=800):

    height, width = frame.shape[:2]

    # Görüntü zaten hedef genişlikten küçükse değiştirme
    if width <= target_width:
        return frame

    # Yeni genişliğe göre küçültme oranını hesapla
    scale_ratio = target_width / width

    # En-boy oranını koruyarak yeni yüksekliği hesapla
    target_height = int(height * scale_ratio)

    # INTER_AREA:
    # Görüntü küçültme işlemlerinde uygun interpolation yöntemidir
    return cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )

capture = cv2.VideoCapture("videos/takip.mp4")

if not capture.isOpened():
    raise RuntimeError("video acilmadi")

# videonun ilk karesini oku
success, previous_frame = capture.read()

if not success:
    raise RuntimeError("ilk kare okunamadi")

# video resize
previous_frame = resize_frame(previous_frame, target_width=800)

# optical flow hesabı tek kanallı gri seviye görüntüler üzerinden yapılır
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# Optical flowu renkli göstermek için video karesiyle aynı boyutta boş hsv görüntüsü oluştur.
hsv = np.zeros_like(previous_frame)

# saturation kanalını maksimum yap,
# böylece farklı hareket yönkeri canlı renklerle gösterilir.
hsv[..., 1] = 255

while True:

    success, frame = capture.read()

    if not success: break

    # bütün kareleri ilk kare ile aynı ölçeğe getir
    frame = resize_frame(frame, target_width=800)

    # mevcut kareyi gri seviyeye dönüştür
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical flow farneback 
    # pyr_scale: görüntü piramidinde her yeni seviye yarı boyuta indirgenir
    # winsize: hareket hesabında kullanılan komşuluk pencere boyutu
    # poly_n: piksel komşuluğunu modellemek için kullanılan alanın boyutu
    # poly_sigma: polinom yaklaşımı öncesinde gaussian yumuşatma miktarı
    # flags: ek bir farneback seçeneği kullanılmıyor
    flow = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, 
                                        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    # x ve y hareket bileşenlerini büyüklük ve yön bilgisine dönüştürür.
    # magnitude: hareketin ne kadar güçlü olduğu
    # angle: hareketin hangi yçne doğru olduğu 
    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])

    # hareket yönünü hsv görüntüsünün Hue kanalında göster
    # opencv de hue değerleri 0-180 aralığında olduğu için açıya dönüştür
    hsv[..., 0] = angle * 180 / np.pi / 2

    # hareket büyüklüğünü 0-255 aralığına normalize et
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # HSV hareket görüntüsünü ekranda göstermek için BGR'a dönüştür
    flow_visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("orijinal video", frame)
    cv2.imshow("farneback optical flow", flow_visualization)

    previous_gray = current_gray.copy()

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()