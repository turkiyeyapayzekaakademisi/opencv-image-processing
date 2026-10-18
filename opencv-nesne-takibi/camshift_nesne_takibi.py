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

# camshift algoritması başlangıç arama penceresi oluştur
track_window = (x, y, width, height)

# ilk kareden yalnızca seçilen nesne bölgesini al
roi = first_frame[y: y+height, x:x+width]

# BGR görüntüyü HSV renk uzayına dönüştür
# camshift hue kanalını kullanacak
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# çok düşük doygunluk ve parlaklıktaki pikselleri histogramdan çıkarmak için maske oluştur
# H: 0-180 bütün hue aralığı
# S: 60-255 çok düşük doygunluk bölgelerini eler
# V: 32-255 çok karanlık bölgeleri eler
roi_mask = cv2.inRange(hsv_roi, np.array((0, 60, 32)), np.array((180, 255, 255)))

# takip edilecek nesnenin hue histogramını hesapla
# [0]: HSV nin 0. kanalı yani Hue
# roi_mask: sadece maskenin izin verdiği pikseller histogram da kullanılır
# [180]: hue histogram bin sayısı
# [0, 180]: hue aralığı
roi_hist = cv2.calcHist([hsv_roi], [0], roi_mask, [180], [0, 180])

# Histogram değerini 0-255 arasında ölçeklendir
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# cam shift arama kriterini ne zaman durdurur
# eps: pencere hareketi yeterince küçüldüğünde
# count: maksimum iterason sayısına ulaştığında
# 10: maksimum iterasyon sayısı
# 1: pencere hareketi 1 pikselden küçük olursa yakınsamış kabul edilir
term_criteria = (
    cv2.TERM_CRITERIA_EPS
    | cv2.TERM_CRITERIA_COUNT,
    10, 1 
)

while True:

    success, frame = capture.read()

    if not success:
        break

    frame = resize_frame(frame, target_width=800)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    back_projection = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # cam shift
    rotated_rect, track_window = cv2.CamShift(back_projection, track_window, term_criteria)

    # box point: camshifting rotated_rect sonucunu 4 köşe olarak verirs
    box_points = cv2.boxPoints(rotated_rect)

    box_points = box_points.astype(np.int32)

    # döndürülmüş takip kutusunu görüntü üzerine çizdir
    cv2.polylines(frame, [box_points], True, (0, 255, 0), 2)

    # takip edilen nesnenin merkez noktası
    center_x = int(rotated_rect[0][0])
    center_y = int(rotated_rect[0][1])

    # merkez noktasına kırmızı daire
    cv2.circle(frame, (center_x, center_y), 5, (0,0,255),-1)

    cv2.imshow("camshift", frame)

    cv2.imshow("histogram back projection", back_projection)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()

cv2.destroyAllWindows()
