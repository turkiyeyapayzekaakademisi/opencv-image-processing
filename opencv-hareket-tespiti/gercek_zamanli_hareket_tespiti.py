import cv2

capture = cv2.VideoCapture("videos/hareket_2.mp4")

if not capture.isOpened():
    raise RuntimeError("video dosyası açılmadı")

# videonun fps değeri
fps = capture.get(cv2.CAP_PROP_FPS) 

# eğer fps geçerliyse bir fram'in yaklaşık ekranda kalma süresini ms cinsinden hesapla
# 25 fps > 1000 /24 = 40 ms
delay = int(1000/ fps) if fps > 0 else 30

# arka plan çıkarma
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=64, detectShadows=True)

# hareket maskesini temizlemek için 5x5 lik eliptik kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

min_area = 1000

while True:

    success, frame = capture.read()

    if not success:
        break

    # arka plan modeli üzerinde ham foreground maskesi oluştur
    foreground_mask = background_subtractor.apply(frame)

    _, motion_mask = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)

    # opening ve closing
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # temizlenmiş maskede dış konturları bul
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # karede kabul edilen anlamlı hareket bölgesinin sayısı
    motion_count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        cv2.rectangle(frame, (x,y), (x + width, y + height), (0, 255, 0), 3)

        motion_count += 1

    # en az 1 tane anlamlı bölge varsa hareket var kabul et
    motion_detected = motion_count > 0

    status_text = "Hareket var" if motion_detected else "Hareket yok"

    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if motion_detected else (0, 255, 0), 2)

    cv2.putText(frame, f"hareket bolgesi: {motion_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("gercek zamanli hareket tespiti", frame)
    cv2.imshow("hareket maskesi", motion_mask)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()