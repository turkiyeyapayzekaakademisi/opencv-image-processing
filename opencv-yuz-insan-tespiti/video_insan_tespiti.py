import cv2

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    raise RuntimeError("kamera acilmadi")

# histogram of oriented gradients özelliklerini hesaplayan bir opencv sınıfı
hog = cv2.HOGDescriptor()

# svm modeli katsayıları
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:

    success, frame = capture.read()

    if not success:
        break

    frame = cv2.flip(frame, 1) # ayna görünümü oluşturma

    target_width = 640

    if frame.shape[1] > target_width:

        scale_ratio = target_width / frame.shape[1]

        target_height = int(frame.shape[0] * scale_ratio)

        frame = cv2.resize(frame, (target_width, target_height))

    # winStride: kayan pencerenin görüntü üzerinde her adımda ne kadar ilerleyeceğini belirler.
    boxes, weights = hog.detectMultiScale(frame, hitThreshold=0, winStride=(8,8), padding=(8,8), scale=1.05, groupThreshold=2)

    for (x, y, width, height), weight in zip(boxes, weights):

        cv2.rectangle(frame, (x, y), (x + width, y + height),(0,255,0),2)

        confidence = float(weight)

        cv2.putText(frame, f"confidence: {confidence:.2f}", (x, max(y-10,20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("gercek zamanli insan tespiti", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()

cv2.destroyAllWindows()
