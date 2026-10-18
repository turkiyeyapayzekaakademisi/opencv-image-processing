import cv2
import numpy as np

# video karelerini en-boy oranı koruyarak küçültür
def resize_frame(frame, target_width = 800):

    height, width = frame.shape[:2]

    if width <= target_width:
        return frame

    scale_ratio = target_width / width
    target_height = int(height*scale_ratio)

    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

# hareketli nesneleri bul ve bounding box listesi döndür
def detect_moving_objects(frame, background_subtractor, kernel, min_area = 900):

    # arka plan modelini kullanarak foreground maskesi oluştur
    foreground_mask = background_subtractor.apply(frame)

    # MOG2 gölgeleri yaklaşık 127 olarak üretir, 200 eşiği ile gölge bölgeleri büyük ölçüde temizlenir
    _, motion_mask = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)

    # MORPH_OPEN: küçük beyaz gürültüleri temizle
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # MORPH_CLOSE: hareketli nesne içerisindeki küçük boşlukları kapatır
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # temizlenmiş maskede hareketli bölgelerin konturlarını bul
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # kutular
    boxes = []

    for contour in contours:

        if cv2.contourArea(contour) < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        boxes.append((x, y, width, height))

    return boxes, motion_mask

class CentroidTracker:

    def __init__(self, max_disappeared=15, max_distance=60):

        # Yeni nesnelere verilecek sıradaki ID
        self.next_object_id = 0

        # object_id -> centroid
        self.objects = {}

        # Her nesnenin kaç karedir görülmediğini tutar
        self.disappeared = {}

        # Nesne bu kadar kare görülmezse takipten çıkarılır
        self.max_disappeared = max_disappeared

        # İki centroid aynı nesne kabul edilirken izin verilen maksimum mesafe
        self.max_distance = max_distance


    def register(self, centroid):

        object_id = self.next_object_id

        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0

        self.next_object_id += 1

        return object_id


    def deregister(self, object_id):

        # Uzun süredir görülmeyen nesneyi takip sisteminden çıkar
        del self.objects[object_id]
        del self.disappeared[object_id]


    def update(self, boxes):

        # Mevcut karede hiçbir nesne bulunmadıysa
        if len(boxes) == 0:

            for object_id in list(self.disappeared.keys()):

                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return {}


        # Her bounding box için centroid hesapla
        input_centroids = np.zeros((len(boxes), 2), dtype="int")

        for i, (x, y, width, height) in enumerate(boxes):

            center_x = x + width // 2
            center_y = y + height // 2

            input_centroids[i] = (center_x, center_y)


        # Henüz takip edilen nesne yoksa hepsini yeni ID ile kaydet
        if len(self.objects) == 0:

            visible_objects = {}

            for centroid in input_centroids:

                centroid = tuple(centroid)
                object_id = self.register(centroid)

                visible_objects[object_id] = centroid

            return visible_objects


        # Önceki nesnelerin ID ve centroid bilgilerini al
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))


        # Her eski centroid ile her yeni centroid arasındaki
        # Öklid mesafesini hesapla
        distances = np.linalg.norm(
            object_centroids[:, None] - input_centroids[None, :],
            axis=2
        )


        # En yakın eşleşmeleri önce değerlendirmek için satırları sırala
        rows = distances.min(axis=1).argsort()

        # Her eski centroid için en yakın yeni centroidi bul
        cols = distances.argmin(axis=1)[rows]


        used_rows = set()
        used_cols = set()
        visible_objects = {}


        for row, col in zip(rows, cols):

            # Daha önce kullanılan eşleşmeleri tekrar kullanma
            if row in used_rows or col in used_cols:
                continue

            # Mesafe çok büyükse aynı nesne olarak kabul etme
            if distances[row, col] > self.max_distance:
                continue


            object_id = object_ids[row]
            centroid = tuple(input_centroids[col])


            # Nesnenin yeni centroid konumunu güncelle
            self.objects[object_id] = centroid

            # Nesne tekrar görüldüğü için kayıp sayacını sıfırla
            self.disappeared[object_id] = 0

            visible_objects[object_id] = centroid


            used_rows.add(row)
            used_cols.add(col)


        # Bu karede eşleşmeyen eski nesneleri bul
        unused_rows = set(range(len(object_centroids))) - used_rows

        for row in unused_rows:

            object_id = object_ids[row]

            self.disappeared[object_id] += 1

            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)


        # Hiçbir eski nesneyle eşleşmeyen centroidler yeni nesnedir
        unused_cols = set(range(len(input_centroids))) - used_cols

        for col in unused_cols:

            centroid = tuple(input_centroids[col])

            object_id = self.register(centroid)

            visible_objects[object_id] = centroid


        return visible_objects

def main():

    # video dosyasını aç
    capture = cv2.VideoCapture("videos/sayma.mp4")

    if not capture.isOpened():
        raise RuntimeError("video acilmadi")

    # MOG2: sabit arka plandan hareketli araçları ayırmak için kullanılıyor
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True) 

    # morfolojik işlemlerde kullanılacak bir eliptik kernel oluştur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # centroid takip sistemini oluştur
    tracker =CentroidTracker(max_disappeared=15, max_distance=60)

    while True:

        success, frame = capture.read()

        if not success:
            break

        # büyük videoyu 800 piksel genişliğine küçült
        frame = resize_frame(frame, target_width=800)

        boxes, motion_mask = detect_moving_objects(frame, background_subtractor, kernel, min_area=900)

        # centroidleri kareler arası eşleştir
        objects = tracker.update(boxes)

        # tespit edilen hareketli bölgelerin bounding boxlarını çiz
        for x, y, width, height in boxes:

            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # takip edilen nesnelerin centroid ve id bilgilerini göster
        for object_id, centroid in objects.items():

            center_x, center_y = centroid

            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

            # nesnenin id bilgisini centroid üzerine yazdır
            cv2.putText(frame, f"ID: {object_id}", (center_x - 20, center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("centroid tracking", frame)
        cv2.imshow("hareket maskesi", motion_mask)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()