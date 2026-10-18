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

success, first_frame = capture.read()

if not success:
    raise RuntimeError("Videonun ilk karesi okunamadi")

first_frame = resize_frame(first_frame, target_width=800)

# ROI seçimi: takip edeceğimiz nesnenin seçilmesi
# fromCenter: dikdörtgeni köşeden sürekleyerek seç
# showCrosshair: nişangah gösterimi
x, y, width, height = cv2.selectROI("Takip edilecek nesneyi sec", first_frame, fromCenter=False, showCrosshair=True)

cv2.destroyWindow("Takip edilecek nesneyi sec")

# kullanıcı seçim yapamadıysa
if width == 0 or height == 0:
    capture.release()
    raise RuntimeError("takip edilecek nesne secilemedi")

# meanshift algoritmasının başlangıç arama penceresi
# x, y: sol üst köşe
# width, height: pencere boyutu
track_window = (x, y, width, height)

# ilk kareden yalnızca seçilen nesne bölgesini al
roi = first_frame[y: y + height, x: x + width]

# BGR görüntüyü HSV renk uzayına dönüştür
# meanshift hue kanalını kullanacak
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

# mean shift arama kriterini ne zaman durdurur
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

    # her yeni kareyi hsv renk uzayına dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ilk karede çıkardığımız nesne histogramının yeni görüntünün hangi bölgelerinde yüksek olasılık ile bulunduğunu hesaplar
    back_projection = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # mean shift
    _, track_window = cv2.meanShift(back_projection, track_window, term_criteria)

    # güncellenmiş takip penceresinin koordinatlarını al
    x, y, width, height = track_window

    cv2.rectangle(frame, (x, y), (x + width, y+ height), (0, 255, 0), 2)

    cv2.imshow("meanshift nesne takibi", frame)

    cv2.imshow("histogram back_projection", back_projection)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()

"""
ilk kareyi al, nesneyi roi olarak seç, hsv uzayına dön, hue histogram hesapla, 
yeni kare gelsin, histogram backprojection, meanshift, yeni nesne konumu bul
"""