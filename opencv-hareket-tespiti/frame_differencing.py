import cv2

capture = cv2.VideoCapture("videos/hareket_2.mp4")

if not capture.isOpened():
    raise RuntimeError("video dosyasi acilmadi")

# ilk kareyi oku: sonraki kareler ile karsilastirmak için referans olarak kullan
success, previos_frame = capture.read()

if not success:
    raise RuntimeError("videodan ilk kare okunamadi")

previous_gray = cv2.cvtColor(previos_frame, cv2.COLOR_BGR2GRAY)

# gürültü azaltma
previous_gray = cv2.GaussianBlur(previous_gray, (5, 5), 0)

while True:

    success, frame = capture.read()

    if not success:
        break

    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)

    # fark bulma
    # absdiff: iki görüntünün aynı frame de ki mutlak piksel farkları
    # fark yüksek ise o bölgede iki kare arasında hareket var
    frame_difference = cv2.absdiff(previous_gray, current_gray)

    # filtreleme
    # 25: hareket kabul edilecek min piksel farkı
    # 255 = beyaz
    _, motion_mask = cv2.threshold(frame_difference, 100, 255, cv2.THRESH_BINARY)

    # maskede ki küçük kopuk bölgeleri biraz daha genişleterek hareketi belirgin hale getir
    # None: 3x3 filtre
    # iterations = 2 ise dilation işleminin 2 kez uygulanması
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)

    cv2.imshow("video", frame)
    cv2.imshow("kare farki", frame_difference)
    cv2.imshow("hareket maskesi", motion_mask)

    # mevcut kareyi bir sonraki döngü için önceki kare yap
    previos_gray = current_gray

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()    


    