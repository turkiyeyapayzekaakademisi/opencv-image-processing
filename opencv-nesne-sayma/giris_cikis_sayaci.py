import cv2
from centroid_tracker import CentroidTracker, detect_moving_objects, resize_frame


# Nesnenin x koordinatına göre bulunduğu bölgeyi belirler
def get_region(center_x, zone_left, zone_right):

    if center_x < zone_left:
        return "left"

    if center_x > zone_right:
        return "right"

    return "inside"


# Video dosyasını aç
capture = cv2.VideoCapture("videos/sayma.mp4")

if not capture.isOpened():
    raise RuntimeError("Video acilmadi")


# MOG2 ile hareketli araçları arka plandan ayır
background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=16, detectShadows=True
)


# Hareket maskesini temizlemek için kernel oluştur
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# Centroid takip sistemi oluştur
tracker = CentroidTracker(max_disappeared=15, max_distance=60)


# Her Object ID'nin geçiş durumunu saklar
object_states = {}


# Giriş ve çıkış sayaçları
entry_count = 0
exit_count = 0


while True:

    success, frame = capture.read()

    if not success:
        break


    # Videoyu küçült
    frame = resize_frame(frame, target_width=800)

    height, width = frame.shape[:2]


    # Sayım çizgisini görüntünün ortasına yerleştir
    line_x = int(width / 2)

    # Sayım çizgisinin sağında ve solunda geçiş bölgesi oluştur
    zone_margin = 50
    zone_left = line_x - zone_margin
    zone_right = line_x + zone_margin


    # Hareketli araçların bounding box'larını bul
    boxes, motion_mask = detect_moving_objects(
        frame, background_subtractor, kernel, min_area=900
    )


    # Araçların centroidlerini Object ID ile takip et
    objects = tracker.update(boxes)


    # Sayım bölgesini saydam olarak göster
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (zone_left, 0),
        (zone_right, height),
        (255, 255, 0),
        -1
    )

    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)


    # Sayım bölgesi sınırlarını çiz
    cv2.line(frame, (zone_left, 0), (zone_left, height), (255, 0, 0), 2)
    cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 2)
    cv2.line(frame, (zone_right, 0), (zone_right, height), (255, 0, 0), 2)


    # Araçların bounding box'larını çiz
    for x, y, box_width, box_height in boxes:

        cv2.rectangle(
            frame,
            (x, y),
            (x + box_width, y + box_height),
            (0, 255, 0),
            2
        )


    # Takip edilen araçların geçişlerini kontrol et
    for object_id, centroid in objects.items():

        center_x, center_y = centroid


        # Aracın mevcut bölgesini belirle
        current_region = get_region(
            center_x,
            zone_left,
            zone_right
        )


        # Nesne ilk kez görülüyorsa durum oluştur
        if object_id not in object_states:

            object_states[object_id] = {
                "last_region": current_region,
                "entry_side": None
            }


        state = object_states[object_id]
        previous_region = state["last_region"]


        # Nesne dışarıdan sayım bölgesine girdi
        if previous_region != "inside" and current_region == "inside":

            # Sayım bölgesine hangi taraftan girdiğini kaydet
            state["entry_side"] = previous_region


        # Nesne sayım bölgesinden çıktı
        elif previous_region == "inside" and current_region != "inside":

            # Soldan girip sağdan çıktıysa giriş
            if state["entry_side"] == "left" and current_region == "right":

                entry_count += 1


            # Sağdan girip soldan çıktıysa çıkış
            elif state["entry_side"] == "right" and current_region == "left":

                exit_count += 1


            # Geçiş tamamlandıktan sonra giriş tarafını sıfırla
            state["entry_side"] = None


        # Sonraki kare için mevcut bölgeyi kaydet
        state["last_region"] = current_region


        # Centroid noktasını göster
        cv2.circle(frame, centroid, 5, (0, 0, 255), -1)


        # Object ID bilgisini göster
        cv2.putText(
            frame,
            f"ID {object_id}",
            (center_x - 20, center_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )


    # Tracker'dan tamamen silinen nesnelerin durumlarını temizle
    active_ids = set(tracker.objects.keys())

    for object_id in list(object_states.keys()):

        if object_id not in active_ids:
            del object_states[object_id]


    # Giriş ve çıkış sayılarını ekranda göster
    cv2.putText(
        frame,
        f"Giris: {entry_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Cikis: {exit_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


    cv2.imshow("Giris Cikis Sayaci", frame)
    cv2.imshow("Hareket Maskesi", motion_mask)


    if cv2.waitKey(25) & 0xFF == ord("q"):
        break


capture.release()
cv2.destroyAllWindows()