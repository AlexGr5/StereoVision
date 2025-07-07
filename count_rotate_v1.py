import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
import math
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

def match_descriptors(desc1, desc2, ratio_thresh=0.6):
    """Векторизованное сопоставление дескрипторов"""
    matches = []
    for i, d1 in enumerate(desc1):
        dists = np.sqrt(np.sum((desc2 - d1)**2, axis=1))
        idx_sorted = np.argsort(dists)
        if dists[idx_sorted[0]] < ratio_thresh * dists[idx_sorted[1]]:
            matches.append((i, idx_sorted[0]))
    return matches

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
    corners = non_max_suppression(R, threshold_ratio=0.05, window_size=3, border_size=border)
    selected_points = anms(corners, num_points)
    
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

def find_top_bottom_points(points):
    """Определение верхней и нижней точек"""
    sorted_points = sorted(points, key=lambda x: x[0])
    return sorted_points[0], sorted_points[-1]

def calculate_angle(point1, point2):
    """Вычисление угла между горизонталью и прямой между двумя точками"""
    dy = point2[0] - point1[0]
    dx = point2[1] - point1[1]
    if dx == 0:
        return 90.0
    angle_rad = math.atan2(dy, dx)  # Учитывает все 4 четверти
    return math.degrees(angle_rad)

def save_mediana(mediana, output_name):
    with open(output_name, "w") as file:
        file.write(str(mediana))
        print(f'\nМедиана сохранена! Результат в {output_name}')

def main():
    captures_dir = "captures"
    aligned_dir = "aligned_rotated"
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
    
    rotation_angles = []  # Список для хранения углов поворота
    
    # Обработка каждой пары изображений
    for i, (left_path, right_path) in enumerate(pairs):
        try:
            print(f"\nОбработка пары {i+1}/{len(pairs)}: {os.path.basename(left_path)}")
            
            # Обработка изображений
            points1, desc1, scale1 = process_image(left_path, num_points=10)
            points2, desc2, scale2 = process_image(right_path, num_points=10)
            
            # Сопоставление дескрипторов
            matches = match_descriptors(desc1, desc2, ratio_thresh=0.5)
            if len(matches) < 2:
                print("Не найдено достаточного количества соответствий для вычисления угла")
                continue
            
            # Получение соответствующих точек
            #matched_points1 = [points1[idx] for idx, _ in matches[:2]]
            #matched_points2 = [points2[idx] for _, idx in matches[:2]]
            matches_sorted = sorted(matches, key=lambda m: np.linalg.norm(desc1[m[0]] - desc2[m[1]]))
            matched_points1 = [points1[i] for i, _ in matches_sorted[:2]]
            matched_points2 = [points2[i] for _, i in matches_sorted[:2]]
            
            # Нахождение верхней и нижней точек
            top1, bottom1 = find_top_bottom_points(matched_points1)
            top2, bottom2 = find_top_bottom_points(matched_points2)
            
            # Вычисление углов
            angle_left = calculate_angle(top1, bottom1)
            angle_right = calculate_angle(top2, bottom2)
            
            # Вычисление разницы в углах
            angle_diff = angle_left - angle_right
            rotation_angles.append(angle_diff)  # Сохранение угла поворота
            
            print(f"Угол наклона на левом изображении: {angle_left:.2f}°")
            print(f"Угол наклона на правом изображении: {angle_right:.2f}°")
            print(f"Необходимый угол поворота правого изображения: {angle_diff:.2f}°")
            
            # Поворот правого изображения
            img_right = Image.open(right_path)
            rotated_img = img_right.rotate(angle_diff, expand=True, resample=Image.BICUBIC)
            
            # Сохранение результата
            right_name = os.path.basename(right_path)
            rotated_img.save(os.path.join(aligned_dir, f"rotated_{right_name}"))
            print(f"Изображение сохранено: {os.path.join(aligned_dir, f'rotated_{right_name}')}")
            
            # Визуализация результатов
            plt.figure(figsize=(12, 6))
            
            # Отображение левого изображения
            plt.subplot(1, 2, 1)
            img_left = Image.open(left_path)
            plt.imshow(img_left, cmap='gray')
            plt.plot([top1[1], bottom1[1]], [top1[0], bottom1[0]], 'r-', linewidth=2)
            plt.title(f'Левое изобр. Угол: {angle_left:.2f}°')
            plt.axis('off')
            
            # Отображение правого изображения после поворота
            plt.subplot(1, 2, 2)
            plt.imshow(rotated_img, cmap='gray')
            plt.plot([top2[1], bottom2[1]], [top2[0], bottom2[0]], 'r-', linewidth=2)
            plt.title(f'Правое изобр. После поворота\nУгол: {angle_right + angle_diff:.2f}°')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(aligned_dir, f"visualization_{i}.png"))
            plt.close()
            
        except Exception as e:
            print(f"Ошибка при обработке {left_path}: {str(e)}")
    
    # Вычисление статистики углов поворота
    if rotation_angles:
        median_angle = np.median(rotation_angles)
        mean_angle = np.mean(rotation_angles)
        std_angle = np.std(rotation_angles)
        
        print(f"\nСтатистика углов поворота:")
        print(f"Медиана: {median_angle:.2f}°")
        print(f"Среднее: {mean_angle:.2f}°")
        print(f"Стандартное отклонение: {std_angle:.2f}°")
        print(f"Общее количество пар: {len(rotation_angles)}")
        
        # Визуализация распределения углов поворота
        plt.figure(figsize=(10, 6))
        plt.hist(rotation_angles, bins=20, alpha=0.7, color='blue')
        plt.axvline(median_angle, color='red', linestyle='dashed', linewidth=2, 
                   label=f'Медиана: {median_angle:.2f}°')
        plt.axvline(mean_angle, color='green', linestyle='dashed', linewidth=2, 
                   label=f'Среднее: {mean_angle:.2f}°')
        plt.xlabel('Угол поворота (градусы)')
        plt.ylabel('Количество пар')
        plt.title('Распределение углов поворота')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(aligned_dir, 'rotation_distribution.png'))
        #plt.close()
        print(f"График распределения углов поворота сохранен в {aligned_dir}/rotation_distribution.png")
        
        save_mediana(median_angle, r'output/rotate.txt')
    else:
        print("\nНе удалось вычислить углы поворота для всех пар изображений")
    
    print("\nОбработка завершена. Результаты сохранены в папке 'aligned_rotated'.")
    
if __name__ == "__main__":
    main()