import cv2

# video karesini en-boy oranını koruyarak küçültür
def resize_frame(frame, target_width = 800):

    height, width = frame.shape[:2]

    if width <= target_width:
        return frame

    scale_ratio = target_width / width
    target_height = int(height * scale_ratio)

    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

# video dosyasını aç
capture = cv2.VideoCapture("videos/coklu_takip.mp4")
if not capture.isOpened():
    raise RuntimeError("video acilmadi")

# ilk kareyi oku
success, first_frame = capture.read()

if not success:
    raise RuntimeError("ilk kare okunamadi")

# resize
first_frame = resize_frame(first_frame, target_width=800)

# takip edilecek nesneyi seç 
bounding_box = cv2.selectROI("takip edilecek nesneyi sec", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("takip edilecek nesneyi sec")

# geçerli roi seçildi mi seçilmedi mi
x, y, width, height = bounding_box

if width == 0 or height == 0:
    capture.release()
    raise RuntimeError("takip edilecek nesne secilmedi")

# CSRT tracker olustur
tracker = cv2.TrackerCSRT_create()

# init: tracker başlatmak
tracker.init(first_frame, bounding_box)

while True:

    success, frame = capture.read()

    if not success:
        break

    frame = resize_frame(frame, target_width=800)

    # update track
    # nesnenin mevcut karadeki yeni bounding box konumunu hesaplar
    tracking_success, bounding_box = tracker.update(frame)

    if tracking_success:

        x, y, width, height = [int(value) for value in bounding_box]

        cv2.rectangle(frame, (x,y), (x+width, y+height),(0, 255, 0), 2)
        cv2.putText(frame, "CSRT - TAKIP", (20,40),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2)

    else:
        cv2.putText(frame, "NESNE KAYBEDILDI", (20,40),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2)

    cv2.imshow("CSRT Nesne takibi", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
