import cv2

# video karelerini en-boy oranı koruyarak küçültür
def resize_frame(frame, target_width = 800):

    height, width = frame.shape[:2]

    if width <= target_width:
        return frame

    scale_ratio = target_width / width
    target_height = int(height*scale_ratio)

    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

# video capture
capture = cv2.VideoCapture("videos/sayma.mp4")
if not capture.isOpened():
    raise RuntimeError("video acilmadi")

while True:

    success, frame = capture.read()

    if not success:
        break

    frame = resize_frame(frame, target_width=800)

    height, width = frame.shape[:2]

    # sayım çizgisini görüntü yüksekliğinin %50 sine yerleştir
    line_x = int(width*0.5)

    # çizginin üstünde veya altında 50 piksellik geçiş bölgesi oluştur
    zone_margin = 50
    zone_left = line_x - zone_margin
    zone_right = line_x + zone_margin

    # sanal bölgeyi göstermek için görüntü kopyası oluştur
    overlay = frame.copy()

    cv2.rectangle(overlay, (zone_left, 0), (zone_right, height), (255, 255,0), -1)

    # dolu dikdörtgeni video ile saydam şekilde birleştir.
    # 0.15 sanal bölgenin görünürlük oranı
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

    # sayım merkez çizgisini çiz
    cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 1)

    # sanal bölge sol sınırı
    cv2.line(frame, (zone_left, 0), (zone_left, height), (255,0,0), 1)

    # sanal bölge sağ sınırı
    cv2.line(frame, (zone_right, 0), (zone_right, height), (255,0,0), 1)

    cv2.putText(frame, "SAYIM BOLGESI", (zone_left - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("sanal sayim bolgesi", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()