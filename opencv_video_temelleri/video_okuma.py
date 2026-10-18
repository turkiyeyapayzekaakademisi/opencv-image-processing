import cv2

# video dosyası okuma AÇ
capture = cv2.VideoCapture("videos/ornek_video.mp4")

# video kaynağının başarıyla açılıp açılmadığını kontrol eder
if not capture.isOpened():
    raise RuntimeError("video dosyasi acilmadi")

while True:

    # OKU
    # success: kare başarıyla okunduysa true döner ama hata varsa yada video bittiyse false döner
    # frame: okunan görüntü karesi 
    success, frame = capture.read()

    # video sona ulaştığında read() false döner ve while döngü biter
    # KARE VAR MI
    if not success:
        break

    # kareyi bgr dan gray scale e dönüştür
    # KAREYİ İŞLE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # GÖSTER
    cv2.imshow("orijinal", frame)
    cv2.imshow("gray scale", gray)

    # waitKey(25): 25 ms bekler ve klavye girişini kontrol eder
    # 27: ESC tuşunun ASCII kodu
    if cv2.waitKey(25) & 0xFF == 27:
        break

# release(): VideoCapture kaynağını serbest bırakır
# VİDEO KAPAT
capture.release()

cv2.destroyAllWindows()

    

