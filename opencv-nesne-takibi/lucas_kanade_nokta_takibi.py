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
success, first_frame = capture.read()

if not success:
    raise RuntimeError("ilk kare okunamadi")

# video resize
first_frame = resize_frame(first_frame, target_width=800)

# select roi
x, y, width, height = cv2.selectROI("Takip edilecek nesneyi sec", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Takip edilecek nesneyi sec")

# kullanıcı seçim yapamadıysa
if width == 0 or height == 0:
    capture.release()
    raise RuntimeError("takip edilecek nesne secilemedi")

# lucas kanade gri seviye kareler üzerinde çalışır
previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# goodFeaturesToTrack için maske oluştur
# siyah bölgelerde özellik aranmaz, beyaz bölgelerde aranır
roi_mask = np.zeros_like(previous_gray)

roi_mask[y:y+height, x:x+width] = 255

# goodFeaturesToTrack
# ROI içerisinde takip edilmeye uygun güçlü köşe noktaları
# maxCorners: bulunacak maksimum özellik sayısı
# qualityLevel: en güçlü köşeye göre minimum kabul edilebilir kalite oranı
# minDistance: iki özellik noktası arasındaki minimum piksel mesafesi
# mask: yalnızca seçilen nesne roi'sinde özellik arar
# blockSize: köşe hesabında kullanılan komşuluk bölgesi
previous_points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, mask = roi_mask, blockSize=7)

if previous_points is None:
    capture.release()
    raise RuntimeError("takip edilecek ozellik noktasi bulunamadi")

# lukas kanade optical flow parametreleri
# winSize: her noktanın yeni konumunun aranacağı pencere boyutu
# maxLevel: görüntü piramidinde kullanılacak maksimum seviye
# criteria: iteratif hesaplamanın durma koşulları
lk_parameters = dict(
    winSize = (21, 21), 
    maxLevel = 3, 
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# takip edilen noktaların geçmiş yollarını çizmek için siyah görüntü oluştur
trajectory_mask = np.zeros_like(first_frame)

while True:

    success, frame = capture.read()

    if not success:
        break

    frame = resize_frame(frame, target_width=800)

    # lucas kanade için yeni kareyi gri seviyeye dönüştür.
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # önceki karedeki noktaların mevcut karedeki yeni konumlarını hesaplar
    # previous_gray: önceki kare
    # current_gray: mevcut kare
    # None: yeni koordinatları OpenCV hesaplar
    current_points, status, error = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, previous_points, None, **lk_parameters)

    if current_points is None or status is None:
        break

    # status değeri 1 olan noktalar başarıyla takip edilmiştir
    good_new = current_points[status.ravel() == 1]
    good_old = previous_points[status.ravel() == 1]

    # başarıyla takip edilen noktaların eski ve yeni konumlarını çizdir
    for new_point, old_point in zip(good_new, good_old):

        new_x, new_y = new_point.ravel()
        old_x, old_y = old_point.ravel()

        new_position = (int(new_x), int(new_y))
        old_position = (int(old_x), int(old_y))

        # noktanın önceki ve yeni konumu arasına çizgi çiz
        # trajectory mask korunduğu için geçmiş hareketler ekranda kalır
        cv2.line(trajectory_mask, old_position, new_position, (0, 255, 0), 2)

        # noktanın mevcut konumu kırmızı daire ile göster
        cv2.circle(frame, new_position, 4, (0, 0, 255), -1)

    # video karesi ile biriken hareket izlerini birleştir
    result = cv2.add(frame, trajectory_mask)

    # takip edilen aktif nokta syısını görüntü üzerine yazdır
    cv2.putText(result, f"takip edilen nokta: {len(good_new)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("lucas-kanade nokta takibi", result)

    # mevcut kare sondaki iterasyonda önceki kare olacak
    previous_gray = current_gray.copy()

    # yalnızca başarıyla takip edilen noktaları sonraki kareye aktarır
    previous_points = good_new.reshape(-1, 1, 2)

    # takip edilecek hiç nokta kalmadıysa işlemi bitir
    if len(previous_points) == 0:
        break

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()

