import cv2
import numpy as np
from collections import deque 
import datetime

# =========================
# 1. YARDIMCI: Oran koruyarak sığdırma (Letterbox)
# (Bu fonksiyon değişmedi)
# =========================
def fit_image_to_box(img, box_w, box_h):
    if img is None or img.size == 0 or box_w <= 0 or box_h <= 0:
        return np.zeros((max(1, box_h), max(1, box_w), 3), dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(box_w / w, box_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    y0 = (box_h - new_h) // 2
    x0 = (box_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

# =========================
# 2. YENİ YARDIMCI: Köşeleri Sıralama
# (Otomatik algılama için gerekli)
# =========================
def order_corners(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# =========================
# 3. YENİ YARDIMCI: Otomatik Grid Tespiti
# (Her karede çalışacak olan 'algılama' fonksiyonu)
# =========================
def find_grid_contour(image, min_area=5000, approx_coef=0.02):
    """
    Görüntüdeki en büyük dörtgeni (kağıt/grid) bulur.
    Dönüş: 4 köşe noktası (np.array) veya None
    """
    # Gürültüyü azaltmak için görüntüyü küçült (daha hızlı çalışır)
    h, w = image.shape[:2]
    scale = 600.0 / w
    small_img = cv2.resize(image, (600, int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Eşikleme (Thresholding)
    # Otsu iyi çalışmazsa, adaptiveThreshold denenebilir
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Kenarları biraz kalınlaştır
    kernel = np.ones((3, 3), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (min_area * scale * scale): # Alanı ölçeğe göre ayarla
            continue
            
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_coef * peri, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # 4 köşe bulundu.
            # Köşeleri orijinal görüntü boyutuna geri ölçekle
            original_corners = approx.reshape(4, 2).astype(np.float32) / scale
            return order_corners(original_corners)
            
    return None # Dörtgen bulunamadı

# =========================
# 4. YARDIMCI: Grid ve Renk Analizi
# (Bu fonksiyonlar değişmedi)
# =========================
def classify_color_hsv(bgr_color):
    bgr_pixel = np.uint8([[bgr_color]])
    hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_pixel[0][0] 

    if s < 40: 
        if v < 50: return "SIYAH"
        if v > 200: return "BEYAZ"
        return "GRI"
    
    if h < 10 or h > 170: return "KIRMIZI"
    elif h < 25: return "TURUNCU"
    elif h < 35: return "SARI"
    elif h < 85: return "YESIL"
    elif h < 130: return "MAVI"
    elif h < 160: return "MOR"
    else: return "PEMBE"

def analyze_grid(warped_image, grid_state, grid_temp_readings, grid_size=(9, 9)):
    # (Bu fonksiyonun içi önceki kodla tamamen aynı)
    if warped_image is None: return None
    h, w = warped_image.shape[:2]
    rows, cols = grid_size
    cell_h = h // rows; cell_w = w // cols
    vis_grid = warped_image.copy() 
    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            pad = int(cell_w * 0.1) 
            cell_roi = warped_image[y1+pad:y2-pad, x1+pad:x2-pad]
            if cell_roi.size == 0: continue
            avg_bgr = cv2.mean(cell_roi)[:3]
            current_color = classify_color_hsv(avg_bgr) 
            grid_temp_readings[i][j].append(current_color)
            if (len(grid_temp_readings[i][j]) == grid_temp_readings[i][j].maxlen and 
                all(c == current_color for c in grid_temp_readings[i][j])):
                if grid_state[i][j] != current_color:
                    grid_state[i][j] = current_color
                    print(f"Guncellendi: Hucure ({i+1}, {j+1}) -> {current_color}")
            stable_color_name = grid_state[i][j] 
            center_x, center_y = x1 + cell_w // 2, y1 + cell_h // 2
            cv2.putText(vis_grid, stable_color_name, (center_x - 15, center_y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis_grid, stable_color_name, (center_x - 15, center_y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(vis_grid, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return vis_grid

# =========================
# 5. YARDIMCI: Oryantasyon Düzeltme
# (Bu fonksiyon değişmedi)
# =========================
def fix_orientation(img_):
    if img_ is None: return None
    img_ = cv2.flip(img_, 1)
    img_ = cv2.rotate(img_, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_

# =========================
# 6. YARDIMCI: Dosyaya Kaydetme
# (Bu fonksiyon değişmedi)
# =========================
def save_grid_to_file(grid_state, filename="grid_sonuc.txt"):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Grid Renk Durumu - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*40 + "\n\n")
            for i, row in enumerate(grid_state):
                formatted_row = [f"{color:<8}" for color in row]
                f.write(f"Satir {i+1}:  | ".join(formatted_row) + "\n")
                if i < len(grid_state) - 1:
                    f.write("-" * (8 * len(row) + 3 * (len(row) - 1)) + "\n")
        print(f"\nGrid durumu başarıyla '{filename}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"\nDosyaya kaydederken bir hata oluştu: {e}")

# =========================
# 7. ANA ÇALIŞTIRMA BLOĞU
# (Tamamen yeniden yazıldı: Manuel kalibrasyon kaldırıldı, otomatik algılama eklendi)
# =========================
if __name__ == "__main__":
    
    # !!! BURAYI DÜZENLE !!!
    VIDEO_SOURCE = "http://192.168.1.128:4747/video" 
    
    GRID_ROWS = 9
    GRID_COLS = 9
    STABILITY_FRAMES = 5 

    # Düzleştirilmiş hedef boyut (A4 dikey oranına yakın)
    DST_W = 630
    DST_H = int(DST_W * np.sqrt(2)) # ~891. 900'e yuvarlayalım (9'a bölünür)
    DST_H = 900
    DST_SIZE = (DST_W, DST_H)
    dst_pts = np.float32([[0, 0], [DST_W, 0], [DST_W, DST_H], [0, DST_H]])
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise ConnectionError(f"Video akışı başlatılamadı: {VIDEO_SOURCE}")
    print("Video akışına bağlanıldı...")
    print("Otomatik grid algılama başlıyor...")
    print("Kapatmak için 'q' veya 'ESC' tuşuna basın.")

    # Grid durumunu saklayan listeler
    grid_state = [["---" for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    grid_temp_readings = [
        [deque(maxlen=STABILITY_FRAMES) for _ in range(GRID_COLS)] 
        for _ in range(GRID_ROWS)
    ]

    # Pencereler
    WIN_CAM = "Kamera Akisi (Otomatik Algilama)"
    WIN_GRID = "Grid Analizi (Düzleştirilmiş)"
    WIN_DEBUG = "Debug - Threshold" # Algılamanın nasıl çalıştığını görmek için
    
    cv2.namedWindow(WIN_CAM, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow(WIN_GRID, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow(WIN_DEBUG, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    
    cv2.resizeWindow(WIN_CAM, 800, 600)
    cv2.resizeWindow(WIN_GRID, 630, 900)
    cv2.resizeWindow(WIN_DEBUG, 600, 400) 

    # Grid bulunamadığında son başarılı görüntüyü göstermek için
    last_good_vis = np.zeros((DST_H, DST_W, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Üzerine çizim yapmak için ham görüntüyü kopyala
        vis_frame = frame.copy() 
        grid_visualization = None

        # 1. OTOMATİK ALGILAMA
        # find_grid_contour'un debug (thr) görüntüsünü de döndürmesi için güncelledik
        
        # --- `find_grid_contour` iç mantığını buraya taşıyalım (debug için) ---
        h, w = frame.shape[:2]
        scale = 600.0 / w
        small_img = cv2.resize(frame, (600, int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quad_points = None # Bu kare için köşe noktaları
        
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < (2000 * scale * scale): # min alan (ayarlanabilir)
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    original_corners = approx.reshape(4, 2).astype(np.float32) / scale
                    quad_points = order_corners(original_corners)
                    break # En büyüğünü bulduk, döngüden çık
        # --- Algılama mantığı bitti ---

        # 2. ANALİZ VE GÖRSELLEŞTİRME
        if quad_points is not None:
            # GEREKSİNİMİ KARŞILA: "grid in etrafı vs yeşil sembollerle belirtilsin"
            cv2.polylines(vis_frame, [quad_points.astype(int)], True, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Kuş bakışı uygula
            M = cv2.getPerspectiveTransform(quad_points, dst_pts)
            warp = cv2.warpPerspective(frame, M, DST_SIZE)
            
            # Oryantasyonu düzelt
            warp = fix_orientation(warp)
            
            # Grid'i analiz et
            if warp is not None:
                grid_visualization = analyze_grid(
                    warped_image=warp, 
                    grid_state=grid_state, 
                    grid_temp_readings=grid_temp_readings, 
                    grid_size=(GRID_ROWS, GRID_COLS)
                )
                last_good_vis = grid_visualization.copy() # Son başarılıyı kaydet
            else:
                grid_visualization = last_good_vis # Hata varsa son başarılıyı kullan
        else:
            # Grid bulunamadıysa, son başarılı grid görüntüsünü göster
            grid_visualization = last_good_vis

        # 3. PENCERELERİ GÖSTER
        try:
            _, _, w_cam, h_cam = cv2.getWindowImageRect(WIN_CAM)
            _, _, w_grid, h_grid = cv2.getWindowImageRect(WIN_GRID)
            _, _, w_dbg, h_dbg = cv2.getWindowImageRect(WIN_DEBUG)

            show_cam = fit_image_to_box(vis_frame, w_cam, h_cam)
            show_grid = fit_image_to_box(grid_visualization, w_grid, h_grid)
            show_dbg = fit_image_to_box(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR), w_dbg, h_dbg)
            
            cv2.imshow(WIN_CAM, show_cam)
            cv2.imshow(WIN_GRID, show_grid)
            cv2.imshow(WIN_DEBUG, show_dbg) # Hata ayıklama penceresi
            
        except cv2.error:
            print("Pencere kapatıldı.")
            break

        # 4. ÇIKIŞ KONTROLÜ
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): 
            print("Çıkış tuşuna basıldı. Grid durumu kaydediliyor...")
            save_grid_to_file(grid_state, filename="grid_sonuc.txt")
            break

    print("Program sonlandırılıyor.")
    cap.release()
    cv2.destroyAllWindows()