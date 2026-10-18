import cv2

# video dosyasını aç
capture = cv2.VideoCapture("videos/hareket_2.mp4")

if not capture.isOpened():
    raise RuntimeError("video dosyasi acilamadi")

# createBackgroundSubtractorMOG2: Gaussian Mixture tabanlı arka plan modeli oluştur
# history: arka plan modelini belirleyen geçmiş kare sayısı
# varThreshold: pikselin arka plan modeline uyup uymadığını belirleyen hassas bir eşik değeri
# detectShadow: hareketli nesnelerin gölgelerini ayrıca tespit etmeye çalışmak
background_subtractor = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 128, detectShadows=True)

while True:

    success, frame = capture.read()
    if not success:
        break

    foreground_mask = background_subtractor.apply(frame)

    cv2.imshow("video", frame)
    cv2.imshow("mog2 foreground mask", foreground_mask)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()