import cv2

image = cv2.imread("images/bird.jpg")

# görüntünün yükseklik genişlik ve kanal bilgilerini alalım
height, width, channels = image.shape

print(f"Yükseklik: {height}")
print(f"Genişlik: {width}")
print(f"Kanal sayısı: {channels}")

# görüntünün toplam eleman sayısı ve veri tipini göster
print(f"toplam eleman: {image.size}")
print(f"veri tipi: {image.dtype}")

# görüntünün merkez koordinatlarını hesapla
x, y = width // 2, height // 2

# x ve y değerlerini kullanarak merkez pikselin BGR değerlerini oku
blue, green, red = image[y, x]

print(f"Merkezi piksel: {[y, x]}")
print(f"blue: {blue}")
print(f"green: {green}")
print(f"red: {red}")