import cv2

capture = cv2.VideoCapture("videos/hareket_2.mp4")

if not capture.isOpened():
    raise RuntimeError("video dosyasini acilmadi")

# arka plan ayıklama
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=124, detectShadows=True)

# MORPH_ELLIPSE: yuvarlatılmış yapısal eleman üretir
# 5, 5: kernel boyutu
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# bu değerden küçük konturları önemsiz hareket veya gürültü kabul et
min_area = 1000

while True:

    success, frame = capture.read()

    if not success:
        break

    # ham foreground maskesini oluştur
    foreground_mask = background_subtractor.apply(frame)

    # 200: eşik seviyesi 
    _, binary_mask = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)

    # open: erosion + dilation
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # close: dilation + erosion
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # RETR_EXTERNAL: yalnızca en dış
    # CHAIN_APPROX_SIMPLE: gereksiz ara kontur noktalarını saklamaz
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = frame.copy()

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        # boundingRect() konturu kapsayan x, y, genişlik ve yükseklik değerleri
        x, y, width, height = cv2.boundingRect(contour)

        # hareketli bölgenin çevresine yeşil kutu çiz
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 2)

    cv2.imshow("ham mog2 maskesi", foreground_mask)
    cv2.imshow("temizlenmis mask", clean_mask)
    cv2.imshow("hareketli bolgeler", result)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()