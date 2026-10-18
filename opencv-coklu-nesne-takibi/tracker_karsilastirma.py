import cv2
import time


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

# resize
first_frame = resize_frame(first_frame, target_width=800)

# ilk kareden birden fazla nesne seç
boxes = cv2.selectROIs("takip edilecek nesneleri sec", first_frame, fromCenter=False,showCrosshair=True)
cv2.destroyWindow("takip edilecek nesneleri sec")

if len(boxes) == 0:
    capture.release()
    raise RuntimeError("herhangi bir nesne secilmedi")

# csrt ve kcf trackerlarını ayrı listelerde sakla
csrt_trackers = []
kcf_trackers = []

# her seçilen araç için hem csrt hem de kcf tracker oluştur
for box in boxes:

    # selectROIs numpy değerlerini int'e çevir
    bounding_box = tuple(map(int, box))

    # csrt tracker oluştur ve aynı roi ile başlat
    csrt_tracker = cv2.TrackerCSRT_create()
    csrt_tracker.init(first_frame, bounding_box)
    csrt_trackers.append(csrt_tracker)

    # kcf tracker oluştur ve aynı roi ile başlat
    kcf_tracker = cv2.TrackerKCF_create()
    kcf_tracker.init(first_frame, bounding_box)
    kcf_trackers.append(kcf_tracker)

# performans değişkenleri tanımla
# işlenen toplam video karesi
frame_count = 0

# tracker update işlemlerinin toplam çalışma süresi
csrt_total_time = 0
kcf_total_time = 0

# başarılı tracker update çağrılarının sayısı
csrt_success_count = 0
kcf_success_count = 0

# yapılan toplam tracker update çağrısı
csrt_total_attempts = 0
kcf_total_attempts = 0

while True:

    success, frame = capture.read()

    if not success:
        break

    # bütün kareleri roi nin seçildiği ölçeğe getir
    frame = resize_frame(frame, target_width=800)

    frame_count += 1

    # aynı video karesinden iki ayrı görselleştirme oluştur
    csrt_frame = frame.copy()
    kcf_frame = frame.copy()

    # CSRT tracking
    csrt_active = 0
    csrt_frame_time = 0

    for object_id, tracker in enumerate(csrt_trackers):

        start_time = time.perf_counter() # tracker update işleminin süresini ölçer

        tracking_success, bounding_box = tracker.update(frame)

        csrt_frame_time += time.perf_counter() - start_time
        csrt_total_attempts += 1

        # tracker nesneyi bulamadıysa kayıp olarak göster
        if not tracking_success:
            cv2.putText(csrt_frame, f"ID {object_id} KAYIP", (20, 100 + object_id*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
            continue

        # başarılı takip sayısını arttır
        csrt_success_count += 1
        csrt_active += 1

        x, y, width, height = [int(value) for value in bounding_box]

        cv2.rectangle(csrt_frame, (x, y), (x + width, y+ height), (0,255,0),2)
        cv2.putText(csrt_frame, f"ID {object_id}",(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

    csrt_total_time += csrt_frame_time

    # kcf tracking
    kcf_active = 0
    kcf_frame_time = 0.0

    for object_id, tracker in enumerate(kcf_trackers):

        start_time = time.perf_counter()

        tracking_success, bounding_box = tracker.update(frame)

        kcf_frame_time += time.perf_counter() - start_time
        kcf_total_attempts += 1


        if not tracking_success:

            cv2.putText(kcf_frame,f"ID {object_id} KAYIP",(20, 130 + object_id * 25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),2)

            continue


        kcf_success_count += 1
        kcf_active += 1


        x, y, width, height = [int(value) for value in bounding_box]


        cv2.rectangle(kcf_frame,(x, y), (x + width, y + height),(0, 255, 0),2)


        cv2.putText(kcf_frame,f"ID {object_id}", (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0), 2)

    kcf_total_time += kcf_frame_time

    # Anlık performans değerleri
    # bir karede ki bütün csrt ve kcf tracker larının tamamlanma hızını hesapla
    csrt_fps = 1 / csrt_frame_time if csrt_frame_time > 0 else 0
    kcf_fps = 1 / kcf_frame_time if kcf_frame_time > 0 else 0

    # video başlangıcından mevcut kareye kadar update başarı oranı
    csrt_success_rate = csrt_success_count / csrt_total_attempts * 100 if csrt_total_attempts > 0 else 0
    kcf_success_rate = kcf_success_count / kcf_total_attempts * 100 if kcf_total_attempts > 0 else 0

    # csrt bilgilerini göster
    cv2.putText(csrt_frame, "CSRT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),2)
    cv2.putText(csrt_frame, f"FPS: {csrt_fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),2)
    cv2.putText(csrt_frame, f"Basari: %{csrt_success_rate:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),2)
    cv2.putText(csrt_frame, f"Aktif: {csrt_active}/{len(csrt_trackers)}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),2)

    # kcf bilgilerini göster
    cv2.putText(kcf_frame, "KCF", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),2)
    cv2.putText(kcf_frame, f"FPS: {kcf_fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),2)
    cv2.putText(kcf_frame, f"Basari: %{kcf_success_rate:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),2)
    cv2.putText(kcf_frame, f"Aktif: {kcf_active}/{len(kcf_trackers)}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),2)

    # sonuçları yan yana göster
    comparison = cv2.hconcat([csrt_frame, kcf_frame])

    # yan yan görüntü çok geniş olduğu için bunu küçült
    comparison = resize_frame(comparison, target_width=1200)

    cv2.imshow("csrt ve kcf performans karsilastirma", comparison)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# final performans sonuçları
# video boyunca bir karede tüm nesneleri takip etme ortalama hızı
csrt_average_fps = frame_count / csrt_total_time if csrt_total_time > 0 else 0
kcf_average_fps = frame_count / kcf_total_time if kcf_total_time > 0 else 0

# bir karedeki bütün tracker ları update etmenin ortalama süresi
csrt_average_ms = csrt_total_time / frame_count * 1000 if frame_count > 0 else 0
kcf_average_ms = kcf_total_time / frame_count * 1000 if frame_count > 0 else 0

# tracker update başarı oranları
csrt_success_rate = csrt_success_count / csrt_total_attempts * 100 if csrt_total_attempts > 0 else 0
kcf_success_rate = kcf_success_count / kcf_total_attempts * 100 if kcf_total_attempts > 0 else 0

print("\nCSRT")
print("----------------------------------------")
print(f"Ortalama FPS       : {csrt_average_fps:.2f}")
print(f"Ort. Sure          : {csrt_average_ms:.2f} ms/frame")
print(f"Takip Denemesi     : {csrt_total_attempts}")
print(f"Basarili Takip     : {csrt_success_count}")
print(f"Takip Basari Orani : %{csrt_success_rate:.2f}")

print("\nKCF")
print("----------------------------------------")
print(f"Ortalama FPS       : {kcf_average_fps:.2f}")
print(f"Ort. Sure          : {kcf_average_ms:.2f} ms/frame")
print(f"Takip Denemesi     : {kcf_total_attempts}")
print(f"Basarili Takip     : {kcf_success_count}")
print(f"Takip Basari Orani : %{kcf_success_rate:.2f}")

capture.release()
cv2.destroyAllWindows()