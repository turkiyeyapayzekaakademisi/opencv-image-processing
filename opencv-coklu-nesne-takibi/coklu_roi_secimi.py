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

# select ROIs
# birinci roi seç, enter, ardından ikinci roi seç, enter ..., tüm roi seçimi tamamlanınca esc
boxes = cv2.selectROIs("takip edilecek nesneleri sec", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("takip edilecek nesneleri sec")

if len(boxes) == 0:
    capture.release()
    raise RuntimeError("herhangi bir nesne secilmedi")

# seçilen bölgeleri göstermek için ilk karenin kopyasını al
preview = first_frame.copy()

for object_id, box in enumerate(boxes):

    x, y, width, height = [int(value) for value in box]

    cv2.rectangle(preview, (x,y), (x +width, y + height),(0, 255, 0), 2)
    cv2.putText(preview, f"ID: {object_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.imshow("secilen nesneler", preview)

cv2.waitKey(0)

capture.release()
cv2.destroyAllWindows()