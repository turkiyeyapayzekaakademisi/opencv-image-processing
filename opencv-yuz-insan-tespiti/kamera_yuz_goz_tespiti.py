import cv2

face_cascade_path = (
    cv2.data.haarcascades 
    + "haarcascade_frontalface_default.xml"
)

eye_cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_eye_tree_eyeglasses.xml"
)

# cascade sınıflandırıcıları yükle
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

if face_cascade.empty() or eye_cascade.empty():
    raise RuntimeError("cascade modelleri yuklenemedi")

# bilgisyar kamerasını aç
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    raise RuntimeError("kamera acilmadi")

while True:
    success, frame = capture.read()

    if not success:
        break

    # kamera görüntüsü yatay çevirme
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # kare içerisinde ki kontrastı iyileştir
    gray = cv2.equalizeHist(gray)

    # kamera karesindeki yüzleri bul
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))

    for x, y, width, height in faces:

        cv2.rectangle(frame, (x,y), (x+width, y+ height),(0,255,0),2)
        # ROI
        face_roi_gray = gray[y: y + height, x: x + width]
    
        face_roi_color = frame[y: y + height, x: x + width]

        # göz tespiti
        eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20,20))

        for eye_x, eye_y, eye_width, eye_height in eyes:
            cv2.rectangle(face_roi_color, (eye_x, eye_y), (eye_x + eye_width, eye_y+ eye_height),(255,0,0),2)

    # karede bulunan yüz sayısını ekrana yazdır
    cv2.putText(frame, f"Yuz sayisi: {len(faces)}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

    cv2.imshow("gercek zamanli yuz ve goz tespiti", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()

