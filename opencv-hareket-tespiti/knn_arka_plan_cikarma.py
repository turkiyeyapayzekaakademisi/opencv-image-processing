import cv2
capture = cv2.VideoCapture("videos/hareket_2.mp4")

if not capture.isOpened():
    raise RuntimeError("video dosyasi acilmadi")

# K en yakın koşu yoklaşımı ile arka plan modeli oluşturur
# dist2Threshold: piksel ile arka plan örnekleri arasındaki kareli mesafe eşiği
background_subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=380, detectShadows=True)

while True:

    success, frame = capture.read()

    if not success:
        break

    foreground_mask = background_subtractor.apply(frame) # hareketli bölgelerin belirtidiği tek kanallı maske

    cv2.imshow("video", frame)
    cv2.imshow("knn foregorund mask", foreground_mask)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()