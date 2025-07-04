"""
Самый лучший вариант программы!
"""

import cv2
import numpy as np
import os
import pickle

# Configuration parameters
LEFT_CAMERA_ID = 1      # ID for left camera
RIGHT_CAMERA_ID = 0     # ID for right camera
CALIBRATION_FILE_LEFT = 'output/calibration_data_left.pkl'  # Calibration file for left camera
CALIBRATION_FILE_RIGHT = 'output/calibration_data_right.pkl' # Calibration file for right camera
DESIRED_RESOLUTIONS = [(1920, 1080), (1280, 720), (640, 480)]  # Priority resolutions
DISPLAY_SIZE = (640, 480)  # Display size

# Константа для вертикального смещения (пиксели)
VERTICAL_SHIFT = 45     # 67 было

# Константа для поворота правого изображения
RIGHT_IMG_ROTATE = 0.0

BASELINE = 0.08         # Расстояние между камерами в метрах
MIN_DISP = 0            # Минимальная диспаратность
NUM_DISP = 16 * 20       # Должно быть кратно 16
WINDOW_SIZE = 5        # Размер окна для совпадений


# Диапазон глубины (в метрах)
min_depth = 0.3     # 0.1
max_depth = 2.0     # 3.0

def set_camera_resolution(cap, desired_resolutions):
    """
    Attempts to set camera resolution from priority list.
    Returns (resolution, success status).
    """
    for width, height in desired_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if (actual_width, actual_height) == (width, height):
            print(f"Resolution set: {width}x{height}")
            return (width, height), True
            
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Using default resolution: {actual_width}x{actual_height}")
    return (actual_width, actual_height), False

def load_calibration_and_init_camera(calibration_file, camera_id):
    """
    Loads calibration data and initializes camera.
    Returns dictionary with calibration data and capture object.
    """
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file not found: {calibration_file}")
        return None
    
    with open(calibration_file, 'rb') as f:
        calibration_data = pickle.load(f)
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return None

    resolution, success = set_camera_resolution(cap, DESIRED_RESOLUTIONS)
    width, height = resolution
    
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['distortion_coefficients']
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width, height), 5)
    
    return {
        'cap': cap,
        'mapx': mapx,
        'mapy': mapy,
        'roi': roi,
        'width': width,
        'height': height,
        'resolution': resolution,
        'calibration_data': calibration_data
    }

def apply_vertical_crop(image, delta_y, is_left_image):
    """
    Добавляет черный пэддинг для вертикального выравнивания изображений
    
    Args:
        image (np.ndarray): Входное изображение (OpenCV BGR)
        delta_y (int): Вертикальное смещение (положительное, если правое ниже)
        is_left_image (bool): True для левого изображения, False для правого

    Returns:
        np.ndarray: Выровненное изображение с черным пэддингом
    """
    height, width = image.shape[:2]
    trim_amount = abs(delta_y)

    if delta_y == 0:
        return image  # Нет смещения — ничего не меняем

    if delta_y > 0:
        # Правое изображение ниже
        if is_left_image:
            # Левое выше — добавляем черный пэддинг снизу
            pad = np.zeros((trim_amount, width, 3), dtype=np.uint8)
            return np.vstack((image, pad))
        else:
            # Правое ниже — добавляем черный пэддинг сверху
            pad = np.zeros((trim_amount, width, 3), dtype=np.uint8)
            return np.vstack((pad, image))
    else:
        # Левое изображение ниже
        if is_left_image:
            # Левое ниже — добавляем черный пэддинг сверху
            pad = np.zeros((trim_amount, width, 3), dtype=np.uint8)
            return np.vstack((pad, image))
        else:
            # Правое выше — добавляем черный пэддинг снизу
            pad = np.zeros((trim_amount, width, 3), dtype=np.uint8)
            return np.vstack((image, pad))

def rotate_image(image, angle):
    """Поворачивает изображение на заданный угол (в градусах) с сохранением исходных размеров"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Получаем матрицу поворота
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Рассчитываем новые размеры с учетом поворота
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # Корректируем матрицу поворота для сохранения центра
    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2
    
    # Выполняем поворот
    rotated = cv2.warpAffine(
        image, 
        M, 
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)  # Черные границы
    )
    
    # Вычисляем координаты для обрезки до исходного размера
    start_x = (new_w - w) // 2
    start_y = (new_h - h) // 2
    end_x = start_x + w
    end_y = start_y + h
    
    # Обрезаем центральную часть
    cropped = rotated[start_y:end_y, start_x:end_x]
    
    # Если из-за округлений размеры не совпали - добавляем черные поля
    if cropped.shape[0] != h or cropped.shape[1] != w:
        result = np.zeros((h, w, 3), dtype=np.uint8)
        y_offset = (h - cropped.shape[0]) // 2
        x_offset = (w - cropped.shape[1]) // 2
        result[y_offset:y_offset+cropped.shape[0], 
               x_offset:x_offset+cropped.shape[1]] = cropped
        return result
    
    return cropped

def main():
    left_data = load_calibration_and_init_camera(CALIBRATION_FILE_LEFT, LEFT_CAMERA_ID)
    right_data = load_calibration_and_init_camera(CALIBRATION_FILE_RIGHT, RIGHT_CAMERA_ID)

    if left_data is None or right_data is None:
        if left_data and left_data['cap'].isOpened():
            left_data['cap'].release()
        if right_data and right_data['cap'].isOpened():
            right_data['cap'].release()
        cv2.destroyAllWindows()
        return

    print(f"\nLeft Camera: {left_data['resolution'][0]}x{left_data['resolution'][1]}")
    print(f"Right Camera: {right_data['resolution'][0]}x{right_data['resolution'][1]}")

    correct_distortion = True
    show_depth = False

    print("\nControls: 'q' - exit, 'd' - toggle distortion correction, 's' - save frame, 'z' - toggle depth map")

    try:
        while True:
            ret_left, frame_left = left_data['cap'].read()
            ret_right, frame_right = right_data['cap'].read()
        
            if not ret_left or not ret_right:
                print("Error: Failed to get frame from camera")
                break
        
            # Применение калибровки
            if correct_distortion:
                undistorted_left = cv2.remap(frame_left, left_data['mapx'], left_data['mapy'], cv2.INTER_LINEAR)
                x, y, w, h = left_data['roi']
                if w > 0 and h > 0:
                    undistorted_left = undistorted_left[y:y+h, x:x+w]
                undistorted_left = cv2.resize(undistorted_left, (left_data['width'], left_data['height']))
        
                undistorted_right = cv2.remap(frame_right, right_data['mapx'], right_data['mapy'], cv2.INTER_LINEAR)
                x, y, w, h = right_data['roi']
                if w > 0 and h > 0:
                    undistorted_right = undistorted_right[y:y+h, x:x+w]
                undistorted_right = cv2.resize(undistorted_right, (right_data['width'], right_data['height']))
        
                # Применяется только при включённой калибровке
                cropped_left = apply_vertical_crop(undistorted_left, VERTICAL_SHIFT, is_left_image=True)
                cropped_right = apply_vertical_crop(undistorted_right, VERTICAL_SHIFT, is_left_image=False)
                
                # Поворот правого изображения
                cropped_right = rotate_image(cropped_right, RIGHT_IMG_ROTATE)
                
                # Агрессивная фильтрация шума
                filtered_left = cv2.bilateralFilter(cropped_left, d=9, sigmaColor=150, sigmaSpace=150)
                filtered_right = cv2.bilateralFilter(cropped_right, d=9, sigmaColor=150, sigmaSpace=150)
                
                # Без фильтрации
                #filtered_left = cropped_left
                #filtered_right = cropped_right
        
            else:
                # Без калибровки — используем оригинальные кадры
                undistorted_left = frame_left
                undistorted_right = frame_right
        
                # Без обрезки и фильтрации
                filtered_left = undistorted_left
                filtered_right = undistorted_right
        
            # Отображение
            display_left = cv2.resize(filtered_left, DISPLAY_SIZE)
            display_right = cv2.resize(filtered_right, DISPLAY_SIZE)
        
            # Статус
            status_text = "Mode: Calibrated + Aligned" if correct_distortion else "Mode: Original"
            cv2.putText(display_left, f"Left Camera | {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_right, f"Right Camera | {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
            combined = np.hstack((display_left, display_right))
            cv2.imshow('Stereo Cameras (q - exit, d - toggle correction, s - save, z - depth map)', combined)
        
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                correct_distortion = not correct_distortion
                print(f"Switched to mode: {'Calibrated + Aligned' if correct_distortion else 'Original'}")
            elif key == ord('s'):
                unique_id = int(cv2.getTickCount())
                os.makedirs("captures", exist_ok=True)
                if correct_distortion:
                    cv2.imwrite(f"captures/left_calibrated_{unique_id}.jpg", filtered_left)
                    cv2.imwrite(f"captures/right_calibrated_{unique_id}.jpg", filtered_right)
                else:
                    cv2.imwrite(f"captures/left_original_{unique_id}.jpg", frame_left)
                    cv2.imwrite(f"captures/right_original_{unique_id}.jpg", frame_right)
                print(f"Frames saved with resolution: {left_data['resolution'][0]}x{left_data['resolution'][1]}")
            elif key == ord('z'):
                show_depth = not show_depth
                print(f"Depth map display: {'On' if show_depth else 'Off'}")
        
            # Карта глубины
            if show_depth:
                gray_left = cv2.cvtColor(filtered_left, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(filtered_right, cv2.COLOR_BGR2GRAY)
        
                # Настройка StereoSGBM
                stereo = cv2.StereoSGBM_create(
                    minDisparity=MIN_DISP,
                    numDisparities=NUM_DISP,
                    blockSize=WINDOW_SIZE,
                    P1=8 * 3 * WINDOW_SIZE**2,
                    P2=32 * 3 * WINDOW_SIZE**2,
                    disp12MaxDiff=1,
                    uniquenessRatio=10,
                    speckleWindowSize=100,
                    speckleRange=32,
                    preFilterCap=63,
                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                )
        
                disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
                
                
                """
                # Другое окружение (py38)
                # pip install opencv-contrib-python
                from cv2.ximgproc import guidedFilter
                guide = cv2.cvtColor(filtered_left, cv2.COLOR_BGR2GRAY)
                disparity = guidedFilter(guide, disparity, radius=15, eps=32)
                """
        
                # Вычисление глубины
                fx = left_data['calibration_data']['camera_matrix'][0, 0]
                depth = (fx * BASELINE) / (disparity + 1e-6)  # Защита от деления на ноль
        
                # Фиксированный диапазон глубины
                depth_clipped = np.clip(depth, min_depth, max_depth)
                depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
        
                # Цветовая карта с плавным градиентом
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
        
                # Отображение
                depth_display = cv2.resize(depth_colormap, DISPLAY_SIZE)
                cv2.imshow('Depth Map', depth_display)
            else:
                if cv2.getWindowProperty('Depth Map', cv2.WND_PROP_VISIBLE) > 0:
                    cv2.destroyWindow('Depth Map')
        
    finally:
        left_data['cap'].release()
        right_data['cap'].release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()