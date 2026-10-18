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

# ilk kareden birden fazla nesne seç
boxes = cv2.selectROIs("takip edilecek nesneleri sec", first_frame, fromCenter=False,showCrosshair=True)
cv2.destroyWindow("takip edilecek nesneleri sec")

if len(boxes) == 0:
    capture.release()
    raise RuntimeError("herhangi bir nesne secilmedi")

# her nesne için ayrı bir tracker oluştur
trackers = []

# ilklendirme
for box in boxes:

    # yeni csrt tracker
    tracker = cv2.TrackerCSRT_create()

    # tracker i ilgili bounding box ile başlat
    tracker.init(first_frame, tuple(box))

    # tracker ı listeye ekle
    trackers.append(tracker)

while True:

    success, frame = capture.read()
    if not success:
        break

    frame = resize_frame(frame, target_width=800)

    # bütün trackerları aynı frame üzerinde güncelle
    for object_id, tracker in enumerate(trackers):

        tracking_success, bounding_box = tracker.update(frame)

        if not tracking_success:
            continue

        x, y, width, height = [int(value) for value in bounding_box]

        cv2.rectangle(frame, (x,y), (x + width, y+ height), (0 ,255,0), 2)
        cv2.putText(frame, f"ID: {object_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("coklu nesne takibi", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()