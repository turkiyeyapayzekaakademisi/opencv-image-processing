import cv2
import numpy as np
from collections import deque

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

# camshift için başlangıç takip penceresi
track_window = (x, y, width, height)

# ilk kareden yalnızca seçilen nesne bölgesini al
roi = first_frame[y:y + height, x:x + width]

# camshift renk histogramı kullanacağı için roi yi HSV'ye dönüştür
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# düşük doygunluk ve çok karanlık pikselleri histogramdan çıkar
roi_mask = cv2.inRange(hsv_roi, np.array((0,60,32)), np.array((180, 255, 255)))

# seçilen nesnenin hue histogramını hesapla
# [0]: hue kanalı
# [180]: histogram bin sayısı
# [0, 180]: hue değer aralığı
roi_hist = cv2.calcHist([hsv_roi], [0], roi_mask, [180], [0, 180])

# histogramı backprojection için 0-255 aralığına normalize et
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# camshift iteratif arama durdurma koşulları
# en fazla 10 iterasyon veya değişim 1 pikselden küçük olduğunda
term_criteria = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    10, 1
)

# deque: nesnenin geçmiş merkez koordinatlarını saklamak için
# maxlen = 100 
trajectory = deque(maxlen=100)

while True:

    success, frame = capture.read()

    if not success:
        break

    # resize
    frame = resize_frame(frame, target_width=800)

    # camshift için mevcut kareyi hsv renk uzayına dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # backprojection: ilk karede oluşturduğumuz nesne histogramının mevcut karede hangi bölgede yoğun olduğunu hesaplar
    back_projection = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # camshift
    rotated_rect, track_window = cv2.CamShift(back_projection, track_window, term_criteria)

    # takip kutusu merkezi hesapla
    center_x = int(rotated_rect[0][0])
    center_y = int(rotated_rect[0][1])

    center = (center_x, center_y)

    trajectory.appendleft(center)

    # camshiftin döndürülmüş takip kutusnu dört köşe noktasına dönüştür
    box_points = cv2.boxPoints(rotated_rect).astype(np.int32)

    # bounding box çiz
    cv2.polylines(frame, [box_points], True, (0, 255, 0), 2)

    # kutunun merkezini kırmızı yap
    cv2.circle(frame, center, 5, (0,0,255),-1)

    # geçmiş mezkez koordinatlarını sırayla birbirne bağla
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i-1], trajectory[i], (0,0,255), 2)

    cv2.putText(frame, f"Merkez: ({center_x}, {center_y})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("nesne hareketi ve izi", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()