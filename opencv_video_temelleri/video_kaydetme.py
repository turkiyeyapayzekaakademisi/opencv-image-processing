import cv2

capture = cv2.VideoCapture("videos/ornek_video.mp4")

if not capture.isOpened():
    raise RuntimeError("video dosyasi acilmadi")

# video writer çıktısının frame boyutu giriş kareleri ile uyumlu olmalı
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# çıktı videosunun oynatma hızını korumak için aynı fps'e ihtiyacımız var
fps = capture.get(cv2.CAP_PROP_FPS)

## video writer
# VideoWriter_fourcc: kullanılacak video codec kodu oluştur
# mp4 formatı
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter("outputs/islenmis_video.mp4", fourcc, fps, (width, height))

if not writer.isOpened():
    capture.release()
    raise RuntimeError("Video writer olusturulamadi")

while True:

    success, frame = capture.read()

    if not success:
        break

    # geçerli kare numarasını al
    frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

    # videoda geçen yaklaşık süre
    elapsed_time = frame_number / fps if fps > 0 else 0

    cv2.putText(frame, f"Kare: {frame_number}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"elapsed_time: {elapsed_time:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,  0.7, (0, 255, 0), 2)

    writer.write(frame)

    # ekranda göster
    cv2.imshow("islenen video", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
writer.release()
cv2.destroyAllWindows()


