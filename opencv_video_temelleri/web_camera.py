import cv2

# 0: ilk veya dahili kamera
# harici başka bir kamera varsa bu da 1-2 gibi indekslerde olabilir
capture = cv2.VideoCapture(0)

# kamera kaynağının başarıyla açıldığını kontrol et
if not capture.isOpened():
    raise RuntimeError("kamera acilmadi")

while True:

    # capture'ı read edelim, eğer kare varsa success = True ve görsel frame içerisinde
    success, frame = capture.read()

    if not success:
        print("kameradan kare alinamadi")
        break

    # görüntüyü belirlenen eksende çevirme
    # 1: yatay çevirme yaparak görüntüyü ayna görünümüne getir
    mirrored = cv2.flip(frame, 1)

    # canlı kamerayı göster
    cv2.imshow("web kamera", mirrored)

    # waitKey(1): canlı akışın hızlı devam etmesi için 1 ms
    # q tuşuna basarsam döngüden çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()