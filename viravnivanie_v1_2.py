import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
import glob
import os

def gaussian_kernel(size=5, sigma=1.0):
    """Генерация 2D гауссова ядра"""
    ax = np.arange(-size//2 + 1, size//2 + 1)
    kernel_1d = np.exp(-ax**2 / (2*sigma**2))
    kernel_1d /= kernel_1d.sum()
    return np.outer(kernel_1d, kernel_1d)

def non_max_suppression(R, threshold_ratio=0.1, window_size=3, border_size=20):
    """Эффективный поиск локальных максимумов"""
    threshold = threshold_ratio * R.max()
    max_filtered = maximum_filter(R, size=window_size, mode='constant')
    mask = (R == max_filtered) & (R >= threshold)
    mask[:border_size] = mask[-border_size:] = False
    mask[:, :border_size] = mask[:, -border_size:] = False
    y, x = np.where(mask)
    return list(zip(y, x, R[y, x]))

def anms(corners, num_points):
    """Оптимизированный ANMS с предварительной сортировкой"""
    if not corners:
        return []
    corners_arr = np.array(corners)
    sorted_idx = np.argsort(-corners_arr[:, 2])
    sorted_corners = corners_arr[sorted_idx]
    y, x = sorted_corners[:, 0], sorted_corners[:, 1]
    
    dy = y[:, None] - y
    dx = x[:, None] - x
    dist_sq = dx**2 + dy**2
    np.fill_diagonal(dist_sq, np.inf)
    min_dists_sq = np.min(dist_sq, axis=1)
    
    radii_idx = np.argsort(-min_dists_sq)[:num_points]
    return [(int(y[i]), int(x[i])) for i in radii_idx]

def compute_hog_descriptor(grad_mag, grad_angle, point, cell_size=8, num_bins=8, window_size=16):
    """Векторизованное вычисление HOG-дескриптора"""
    y, x = map(int, map(round, point))
    half = window_size // 2
    
    if (x < half or x >= grad_mag.shape[1] - half or 
        y < half or y >= grad_mag.shape[0] - half):
        return None
        
    mag_patch = grad_mag[y-half:y+half, x-half:x+half]
    ang_patch = grad_angle[y-half:y+half, x-half:x+half]
    
    descriptor = []
    for i in range(0, window_size, cell_size):
        for j in range(0, window_size, cell_size):
            cell_mag = mag_patch[i:i+cell_size, j:j+cell_size].ravel()
            cell_ang = ang_patch[i:i+cell_size, j:j+cell_size].ravel() % 180
            
            bin_idx = (cell_ang * num_bins / 180).astype(int)
            hist = np.zeros(num_bins)
            for k in range(num_bins):
                hist[k] = np.sum(cell_mag[bin_idx == k])
                
            hist_norm = hist / (np.linalg.norm(hist) + 1e-5)
            descriptor.extend(hist_norm)
    
    descriptor = np.array(descriptor)
    return descriptor / (np.linalg.norm(descriptor) + 1e-5)

def process_image(image_path, num_points=100, max_dim=800):
    """Обработка изображения с масштабированием"""
    image = Image.open(image_path).convert('L')
    img_orig = np.array(image, dtype=np.float32) / 255.0
    
    scale = 1.0
    if max_dim:
        scale = min(max_dim / max(img_orig.shape), 1.0) if max_dim else 1.0
        if scale < 1.0:
            new_size = (int(img_orig.shape[1] * scale), int(img_orig.shape[0] * scale))
            img = np.array(image.resize(new_size, Image.LANCZOS), dtype=np.float32) / 255.0
        else:
            img = img_orig.copy()
    else:
        img = img_orig.copy()
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    
    Ix = convolve2d(img, sobel_x, mode='same', boundary='symm')
    Iy = convolve2d(img, sobel_y, mode='same', boundary='symm')
    
    Ix2, Iy2, Ixy = Ix**2, Iy**2, Ix*Iy
    
    gauss = gaussian_kernel(5, sigma=1.5)
    Sx2 = convolve2d(Ix2, gauss, mode='same', boundary='symm')
    Sy2 = convolve2d(Iy2, gauss, mode='same', boundary='symm')
    Sxy = convolve2d(Ixy, gauss, mode='same', boundary='symm')
    
    k = 0.04
    det = Sx2*Sy2 - Sxy**2
    trace = Sx2 + Sy2
    R = det - k*trace**2
    
    border = 20
    corners = non_max_suppression(R, threshold_ratio=0.1, window_size=3, border_size=border)
    selected_points = anms(corners, num_points // 10)
    
    grad_mag = np.sqrt(Ix**2 + Iy**2)
    grad_angle = np.arctan2(Iy, Ix) * 180 / np.pi
    
    descriptors = []
    valid_points = []
    for point in selected_points:
        desc = compute_hog_descriptor(grad_mag, grad_angle, point)
        if desc is not None:
            descriptors.append(desc)
            valid_points.append(point)
    
    return np.array(valid_points), np.array(descriptors), scale

def match_descriptors(desc1, desc2, ratio_thresh=0.7):
    """Векторизованное сопоставление дескрипторов"""
    matches = []
    for i, d1 in enumerate(desc1):
        dists = np.sqrt(np.sum((desc2 - d1)**2, axis=1))
        idx_sorted = np.argsort(dists)
        if dists[idx_sorted[0]] < ratio_thresh * dists[idx_sorted[1]]:
            matches.append((i, idx_sorted[0]))
    return matches

def crop_images(img1, img2, delta_y_real, trim_amount):
    """Обрезка копий изображений с учетом вертикального смещения"""
    img1_copy = img1.copy()
    img2_copy = img2.copy()
    
    if delta_y_real > 0:
        trim_bottom_first = min(trim_amount, img1.height)
        trim_top_second = min(trim_amount, img2.height)
        
        img1_cropped = img1_copy.crop((0, 0, img1.width, img1.height - trim_bottom_first))
        img2_cropped = img2_copy.crop((0, trim_top_second, img2.width, img2.height))
        
    elif delta_y_real < 0:
        trim_top_first = min(trim_amount, img1.height)
        trim_bottom_second = min(trim_amount, img2.height)
        
        img1_cropped = img1_copy.crop((0, trim_top_first, img1.width, img1.height))
        img2_cropped = img2_copy.crop((0, 0, img2.width, img2.height - trim_bottom_second))
        
    else:
        img1_cropped = img1_copy
        img2_cropped = img2_copy
    
    return img1_cropped, img2_cropped

def compute_vertical_offset(left_path, right_path, num_points=100, max_dim=800):
    """Вычисление вертикального смещения для пары изображений"""
    points1, desc1, scale1 = process_image(left_path, num_points, max_dim)
    points2, desc2, scale2 = process_image(right_path, num_points, max_dim)
    
    matches = match_descriptors(desc1, desc2, ratio_thresh=0.7)
    
    if len(matches) == 0:
        return None
        
    matched_points1 = [points1[idx] for idx, _ in matches]
    matched_points2 = [points2[idx] for _, idx in matches]
    
    delta_ys = [p2[0] - p1[0] for p1, p2 in zip(matched_points1, matched_points2)]
    median_delta_y = np.median(delta_ys)
    
    scale_combined = (scale1 + scale2) / 2
    delta_y_real = median_delta_y / scale_combined
    
    return delta_y_real

if __name__ == "__main__":
    captures_dir = "captures"
    aligned_dir = "aligned"
    os.makedirs(aligned_dir, exist_ok=True)
    
    # Сбор всех пар изображений
    left_files = glob.glob(os.path.join(captures_dir, 'left*.jpg'))
    pairs = []
    for left_file in left_files:
        right_file = left_file.replace('left', 'right')
        if os.path.exists(right_file):
            pairs.append((left_file, right_file))
        else:
            print(f"Правый файл не найден: {right_file}")

    if not pairs:
        print("Не найдено пар изображений для обработки.")
        exit()

    # Первый проход: вычисление смещений
    offsets = []
    for left_path, right_path in pairs:
        try:
            offset = compute_vertical_offset(left_path, right_path)
            if offset is not None:
                print(f"Пара {os.path.basename(left_path)}/{os.path.basename(right_path)}: смещение = {offset:.2f}")
                offsets.append(offset)
            else:
                print(f"Не удалось вычислить смещение для {os.path.basename(left_path)}")
        except Exception as e:
            print(f"Ошибка при обработке {left_path}: {str(e)}")

    if not offsets:
        print("Нет данных для вычисления среднего смещения")
        exit()

    # Статистика смещений
    median_offset = np.median(offsets)
    mean_offset = np.mean(offsets)
    std_offset = np.std(offsets)
    print(f"\nСтатистика смещений:")
    print(f"Медиана: {median_offset:.2f} пикселей")
    print(f"Среднее: {mean_offset:.2f} пикселей")
    print(f"Стандартное отклонение: {std_offset:.2f} пикселей")
    print(f"Общее количество пар: {len(offsets)}")

    # Визуализация распределения смещений
    plt.figure(figsize=(10, 6))
    plt.hist(offsets, bins=20, alpha=0.7, color='blue')
    plt.axvline(median_offset, color='red', linestyle='dashed', linewidth=2, label=f'Медиана: {median_offset:.2f}')
    plt.axvline(mean_offset, color='green', linestyle='dashed', linewidth=2, label=f'Среднее: {mean_offset:.2f}')
    plt.xlabel('Вертикальное смещение (пиксели)')
    plt.ylabel('Количество пар')
    plt.title('Распределение вертикальных смещений')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(aligned_dir, 'offset_distribution.png'))
    print(f"График распределения сохранен в {aligned_dir}/offset_distribution.png")

    # Второй проход: обрезка изображений
    trim_amount = abs(int(round(median_offset)))
    print(f"\nИспользуемое смещение для обрезки: {median_offset:.2f} пикселей")
    
    for i, (left_path, right_path) in enumerate(pairs):
        try:
            img_left = Image.open(left_path)
            img_right = Image.open(right_path)
            
            cropped_left, cropped_right = crop_images(img_left, img_right, median_offset, trim_amount)
            
            # Сохранение результатов
            left_name = os.path.basename(left_path)
            right_name = os.path.basename(right_path)
            cropped_left.save(os.path.join(aligned_dir, f"aligned_{left_name}"))
            cropped_right.save(os.path.join(aligned_dir, f"aligned_{right_name}"))
            print(f"Обработана пара {i+1}/{len(pairs)}: {left_name}")
        except Exception as e:
            print(f"Ошибка при обрезке {left_path}: {str(e)}")

    print("\nОбработка завершена. Результаты сохранены в папке 'aligned'.")