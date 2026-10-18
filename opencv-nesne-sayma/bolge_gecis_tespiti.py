import cv2
from centroid_tracker import CentroidTracker, detect_moving_objects, resize_frame

# nesnenin x fonksiyonuna göre hangi bölgede olduğunu belirleyen bir fonksiyon yaz
def get_region(center_x, zone_left, zone_right):

    if center_x < zone_left:
        return "left"

    if center_x > zone_right:
        return "right"

    return "inside"

# video dosyasını aç
capture = cv2.VideoCapture("videos/sayma.mp4")

if not capture.isOpened():
    raise RuntimeError("video acilmadi")

# mog2 ile hareketli araçların arka plandan ayrılması
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# hereket maskesini temizlemek için kernel oluştur
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# centroid takip sistemi oluştur
tracker = CentroidTracker(max_disappeared=15, max_distance=60)

# her object id için geçiş durumunu saklar
object_states = {}

# ekranda son gerçekleşen geçişi göstermek için last event
last_event = "henuz gecis yok"

while True:

    success, frame = capture.read()

    if not success:
        break

    frame = resize_frame(frame, target_width=800)

    height, width = frame.shape[:2]

    # sayım çizgisini görüntü ortasına yerleştir
    line_x = int(width / 2)

    # çizginin sağında ve solunda geçiş bölgesi oluştur
    zone_margin = 50
    zone_left = line_x - zone_margin
    zone_right = line_x + zone_margin

    # hareketli araçların bounding boxlarını bul
    boxes, motion_mask = detect_moving_objects(frame, background_subtractor, kernel, min_area=900)

    # araç centroidlerinin id'lerle takip et
    objects = tracker.update(boxes)

    # sayım bölgesini saydam olarak göstermek için
    overlay = frame.copy()

    cv2.rectangle(overlay, (zone_left, 0), (zone_right, height), (255, 255, 0), -1)

    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

    # sayım bölgesinin sol sınırı
    cv2.line(frame, (zone_left,0), (zone_left, height), (255, 0, 0), 2)

    # merkez sayım çizgisi
    cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255))

    # sayım bölgesinin sağ sınırı
    cv2.line(frame, (zone_right,0), (zone_right, height), (255, 0, 0), 2)

    # takip edilen araçların geçiş durumu
    for object_id, centroid in objects.items():

        center_x, center_y = centroid

        # aracın mevcut bölgesini belirle
        current_region = get_region(center_x, zone_left, zone_right)

        # nesne ilk defa görülüyorsa durum kaydı oluştur
        if object_id not in object_states:
            object_states[object_id] = {
                "last_region": current_region,
                "entry_side": None
            }

        state = object_states[object_id]

        previous_region = state["last_region"]

        # nesne dışarıdan sayım bölgesine girdi
        if previous_region != "inside" and current_region == "inside":

            # bölgeye soldan mı sağdan mı girdi
            state["entry_side"] = previous_region

        elif previous_region == "inside" and current_region != "inside": # nesne sayım bölgesinden çıktıysa

            if state["entry_side"] == "left" and current_region == "right": # soldan girip sağdan çıktı
                last_event = (f"ID {object_id}: SOL -> SAG")

            elif state["entry_side"] == "right" and current_region == "left":
                last_event = (f"ID {object_id}: SAG -> SOL")

            state["entry_side"] = None


        # sonraki kare için mevcut bölgeyi kaydet
        state["last_region"] = current_region

        # centroid noktasını göster
        cv2.circle(frame, centroid, 5, (0,0,255),-1)

        # object id ve bulunduğu bölgeyi göster
        cv2.putText(frame, f"ID: {object_id} - {current_region}", (center_x - 45, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # tracker dan tamamen silinmiş id lerin durum kayıtlarını temizle
    active_ids = set(tracker.objects.keys())

    for object_id in list(object_states.keys()):
        if object_id not in active_ids:
            del object_states[object_id]

    # son gerçekleşen geçisi ekranda göster
    cv2.putText(frame, last_event, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("bolge gecis tespiti", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()

