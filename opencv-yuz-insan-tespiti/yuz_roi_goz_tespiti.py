import cv2

image = cv2.imread("images/yuz.png")

face_cascade_path = ( cv2.data.haarcascades + "haarcascade_frontalface_default.xml" )

eye_cascade_path = ( cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml" )

# yüz tespiti için cascade classifier
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# göz tespiti için ayrı bir cascade classifier oluştur
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

if face_cascade.empty() or eye_cascade.empty():
    raise RuntimeError("cascade modelleri yuklenemedi")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

for x, y, width, height in faces:
    cv2.rectangle(image, (x,y),(x+width, y+height), (0, 255, 0), 2)

    # ROI
    face_roi_gray = gray[y: y + height, x: x + width]

    face_roi_color = image[y: y + height, x: x + width]

    # yalnızca roi içerisindeki gözleri ara
    eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (15,15))

    for eye_x, eye_y, eye_width, eye_height in eyes:

        cv2.rectangle(face_roi_color, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), (255,0,0),2)

cv2.imshow("yuz ve goz tespiti", image)

cv2.imwrite("outputs/yuz_goz_tespiti.jpg",image)

cv2.waitKey(0)
cv2.destroyAllWindows()
