import cv2

# video dosyasını aç
capture = cv2.VideoCapture("videos/ornek_video.mp4")

if not capture.isOpened():
    raise RuntimeError("video dosyasi acilamadi")

fps = int(capture.get(cv2.CAP_PROP_FPS))

while True:

    success, frame = capture.read()

    if not success:
        break

    # CAP_PROP_POS_FRAMES: sıradaki kare pozisyonu
    # frame numarası için kullanılır
    frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

    # geçen süre: işlenen kare sayısı / fps
    elapsed_time = frame_number / fps if fps > 0 else 0

    cv2.putText(frame, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Videonun o anda işlenen kare numarasını göster
    cv2.putText(frame, f"Kare: {frame_number}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Kare numarası ve fps ile hesaplanan gerçek zaman
    cv2.putText(frame, f"Süre: {elapsed_time:.2f} sn", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Video bilgileri", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()