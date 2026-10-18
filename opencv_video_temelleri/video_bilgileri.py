import cv2

# video dosyasını aç
capture = cv2.VideoCapture("videos/ornek_video.mp4")

if not capture.isOpened():
    raise RuntimeError("video dosyasi acilmadi")

# video genişliği
# get(): video capture kaynağından belli bir özelliği okumak için
# CAP_PROP_FRAME_WIDTH: genişlik bilgisi için
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

# video height: video karelerinin piksel cinsinden yüksekliği
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fps: frame per second 
# fps = 100 ne demek? 1 saniye içinde 100 tane görsel var demek
# CAP_PROP_FPS: videonun saniyede kaç kare içerdiği
fps = int(capture.get(cv2.CAP_PROP_FPS))

# video dosyasındaki toplam kare sayısı
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

# video süresi
duration = frame_count / fps if fps > 0 else 0

print(f"width: {width}")
print(f"height: {height}")
print(f"fps: {fps}")
print(f"frame_count: {frame_count}")
print(f"duration: {duration}")

# ilk kareyi okut
success, frame = capture.read()

if success:

    frame_heigth, frame_width = frame.shape[:2]

    print(f"frame_heigth: {frame_heigth}")
    print(f"frame_width: {frame_width}")

    cv2.imshow("ilk kare", frame)

    cv2.waitKey(0)

capture.release()
cv2.destroyAllWindows()