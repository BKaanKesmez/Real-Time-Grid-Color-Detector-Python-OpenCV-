import cv2
import numpy as np

# === GÖRSELİN DOSYA YOLU ===
image_path = "ahmo.jpg"

# Görseli yükle
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

img = cv2.resize(img, (640, 480))
clone = img.copy()
points = []

# === Fareyle nokta seçme ===
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Points", img)

cv2.imshow("Select Points", img)
cv2.setMouseCallback("Select Points", select_point)

print("\n📌 Sırasıyla 4 noktaya tıkla:")
print("1️⃣ Sol Üst  2️⃣ Sağ Üst  3️⃣ Sağ Alt  4️⃣ Sol Alt")

while True:
    cv2.imshow("Select Points", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        cv2.destroyAllWindows()
        exit()
    elif len(points) == 4:
        break

cv2.destroyWindow("Select Points")

# === Perspektif dönüşümü uygula ===
pts1 = np.float32(points)
pts2 = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
bird_eye = cv2.warpPerspective(clone, matrix, (640, 480))

# === Aynalama düzeltmesi ===
bird_eye = cv2.flip(bird_eye, 1)

# === 90 derece sola döndürme ===
bird_eye = cv2.rotate(bird_eye, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow("Original Image", clone)
cv2.imshow("Bird's Eye View (Final)", bird_eye)

cv2.waitKey(0)
cv2.destroyAllWindows()
