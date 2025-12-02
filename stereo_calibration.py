import numpy as np
import cv2
import glob
import os
import pickle
import sys
import re

def get_opencv_version():
    """
    Возвращает версию OpenCV в формате (major, minor, patch)
    
    Возвращает:
        tuple: Версия OpenCV в виде (мажорная, минорная, патч)
    
    Примечание:
        - Необходимо для совместимости разных версий OpenCV
        - В OpenCV 4.x изменился возвращаемый формат функции stereoCalibrate
    """
    version = cv2.__version__
    match = re.match(r'(\d+)\.(\d+)\.(\d+)', version)
    if match:
        return tuple(map(int, match.groups()))
    return (0, 0, 0)

def calibrate(images_dir, left_pattern, right_pattern, output_dir, chessboard_size, square_size):
    """
    Выполняет стереокалибровку двух камер для определения их взаимного положения в пространстве.
    
    Параметры:
        images_dir (str): Директория с изображениями стереопар
        left_pattern (str): Шаблон имен файлов для левых изображений (например, 'left_*.jpg')
        right_pattern (str): Шаблон имен файлов для правых изображений (например, 'right_*.jpg')
        output_dir (str): Директория для сохранения результатов стереокалибровки
        chessboard_size (tuple): Размер шахматной доски (ширина, высота) в углах
        square_size (float): Реальный размер квадрата шахматной доски в сантиметрах
    
    Процесс:
        1. Загрузка данных индивидуальной калибровки каждой камеры
        2. Поиск и сопоставление стереопар изображений
        3. Обнаружение углов шахматной доски на каждой паре
        4. Выполнение стереокалибровки для определения взаимного положения камер
        5. Стереоректификация для выравнивания изображений
        6. Сохранение результатов и создание тестового изображения
    
    Примечание:
        - Требуется предварительная индивидуальная калибровка каждой камеры
        - Шахматная доска должна быть четко видна на обоих изображениях пары
        - Рекомендуется не менее 8-10 успешных стереопар для точной калибровки
    """
    print("\n=== Начало стереокалибровки ===")
    
    # ШАГ 1: Загружаем данные индивидуальной калибровки левой и правой камер
    left_calib_file = os.path.join(output_dir, 'calibration_data_left.pkl')
    right_calib_file = os.path.join(output_dir, 'calibration_data_right.pkl')
    
    # Проверка существования файлов калибровки
    if not os.path.exists(left_calib_file) or not os.path.exists(right_calib_file):
        print("Ошибка: Не найдены файлы индивидуальной калибровки")
        print(f"Проверьте наличие файлов: {left_calib_file} и {right_calib_file}")
        return
    
    # Загрузка данных калибровки из pickle-файлов
    try:
        with open(left_calib_file, 'rb') as f:
            left_data = pickle.load(f)
        with open(right_calib_file, 'rb') as f:
            right_data = pickle.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке калибровочных файлов: {e}")
        return
    
    # Извлечение матриц камер и коэффициентов дисторсии
    mtx_left = left_data['camera_matrix']  # Матрица камеры левой камеры
    dist_left = left_data['distortion_coefficients']  # Коэффициенты дисторсии левой камеры
    mtx_right = right_data['camera_matrix']  # Матрица камеры правой камеры
    dist_right = right_data['distortion_coefficients']  # Коэффициенты дисторсии правой камеры
    
    print("Данные индивидуальной калибровки успешно загружены")
    
    # ШАГ 2: Подготовка объектных точек (3D координаты углов шахматной доски в реальном мире)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  # Масштабируем до реальных размеров (в сантиметрах)
    
    # Массивы для хранения точек для всех изображений
    objpoints = []        # 3D точки в мире (одинаковые для всех изображений)
    left_imgpoints = []   # 2D точки на левых изображениях
    right_imgpoints = []  # 2D точки на правых изображениях
    
    # Получаем списки изображений по шаблонам
    left_images = sorted(glob.glob(os.path.join(images_dir, left_pattern)))
    right_images = sorted(glob.glob(os.path.join(images_dir, right_pattern)))
    
    print(f"Найдено левых изображений: {len(left_images)}")
    print(f"Найдено правых изображений: {len(right_images)}")
    
    # ШАГ 3: Проверяем и сопоставляем пары изображений
    # Ищем соответствие между левыми и правыми изображениями по именам файлов
    valid_pairs = []
    for l_img in left_images:
        base = os.path.basename(l_img).replace('left_', '')  # Убираем префикс 'left_'
        r_img = os.path.join(images_dir, 'right_' + base)    # Формируем имя правого файла
        if os.path.exists(r_img):
            valid_pairs.append((l_img, r_img))  # Добавляем пару, если оба файла существуют
    
    print(f"Найдено {len(valid_pairs)} валидных стереопар")
    
    if len(valid_pairs) == 0:
        print("Ошибка: Не найдено совпадающих пар изображений")
        print("Убедитесь, что имена файлов соответствуют шаблонам left_*.jpg и right_*.jpg")
        return
    
    # Создаем директорию для сохранения изображений с отмеченными углами
    corners_dir = os.path.join(output_dir, 'stereo_corners')
    os.makedirs(corners_dir, exist_ok=True)
    
    # ШАГ 4: Определяем размер изображений (предполагается, что все изображения одинакового размера)
    first_img = cv2.imread(valid_pairs[0][0])
    if first_img is None:
        print("Ошибка: Не удалось загрузить первое изображение для определения размера")
        return
    img_size = first_img.shape[1], first_img.shape[0]  # (ширина, высота)
    print(f"Размер изображений: {img_size}")
    
    # ШАГ 5: Поиск углов шахматной доски на каждой стереопаре
    found_count = 0  # Счетчик успешно найденных пар
    for idx, (l_img_path, r_img_path) in enumerate(valid_pairs):
        print(f"\nОбработка пары {idx+1}/{len(valid_pairs)}:")
        print(f"Левое: {os.path.basename(l_img_path)}")
        print(f"Правое: {os.path.basename(r_img_path)}")
        
        try:
            # Загружаем изображения стереопары
            l_img = cv2.imread(l_img_path)
            r_img = cv2.imread(r_img_path)
            
            if l_img is None or r_img is None:
                print(f"  Ошибка: Не удалось загрузить изображения")
                continue
                
            # Конвертируем в оттенки серого для поиска углов
            gray_l = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)
            
            # Поиск углов шахматной доски с разными флагами для надежности
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)
            
            # Если углы не найдены стандартным методом, пробуем с дополнительными флагами
            if not ret_l or not ret_r:
                # Используем адаптивный порог и нормализацию изображения
                ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, 
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                           cv2.CALIB_CB_NORMALIZE_IMAGE)
                ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, 
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                           cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            # Если углы найдены на обоих изображениях пары
            if ret_l and ret_r:
                found_count += 1
                print(f"  Углы успешно найдены!")
                
                # Уточнение позиций углов с субпиксельной точностью
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
                
                # Сохраняем точки для последующей калибровки
                objpoints.append(objp)  # 3D точки (одинаковые для всех пар)
                left_imgpoints.append(corners_l)  # 2D точки на левом изображении
                right_imgpoints.append(corners_r)  # 2D точки на правом изображении
                
                # Рисуем углы на изображениях для визуальной проверки
                vis_l = l_img.copy()
                vis_r = r_img.copy()
                cv2.drawChessboardCorners(vis_l, chessboard_size, corners_l, ret_l)
                cv2.drawChessboardCorners(vis_r, chessboard_size, corners_r, ret_r)
                
                # Сохраняем изображения с отмеченными углами
                cv2.imwrite(os.path.join(corners_dir, f'left_corners_{idx:03d}.jpg'), vis_l)
                cv2.imwrite(os.path.join(corners_dir, f'right_corners_{idx:03d}.jpg'), vis_r)
            else:
                print("  Углы НЕ найдены на одном или обоих изображениях")
                
        except Exception as e:
            print(f"  Ошибка при обработке пары: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nВсего найдено валидных пар: {found_count} из {len(valid_pairs)}")
    
    # Проверка достаточного количества успешных пар для калибровки
    if found_count < 8:  # Минимум 8 пар для стабильной калибровки
        print(f"Ошибка: Найдено недостаточно валидных пар ({found_count}). Требуется минимум 8.")
        return
    
    print(f"\nНачало стереокалибровки с {found_count} валидными парами...")
    
    # ШАГ 6: Настройка параметров стереокалибровки
    
    # Флаги для стереокалибровки - фиксируем внутренние параметры (они уже известны из индивидуальной калибровки)
    flags = (cv2.CALIB_FIX_INTRINSIC |
             cv2.CALIB_USE_INTRINSIC_GUESS |
             cv2.CALIB_FIX_PRINCIPAL_POINT |
             cv2.CALIB_FIX_FOCAL_LENGTH |
             cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_RATIONAL_MODEL)
    
    # Упрощенные флаги для начала (можно экспериментировать)
    flags = cv2.CALIB_FIX_INTRINSIC  # Фиксируем внутренние параметры камер
    
    # Критерии завершения оптимизации
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    try:
        # ШАГ 7: Выполнение стереокалибровки
        print("Запуск cv2.stereoCalibrate...")
        
        # Определение версии OpenCV для совместимости
        opencv_version = get_opencv_version()
        print(f"Версия OpenCV: {opencv_version}")
        
        # В OpenCV 4.x изменился возвращаемый формат функции stereoCalibrate
        if opencv_version[0] >= 4:
            # Для OpenCV 4.x - возвращает 9 значений
            ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                objpoints, left_imgpoints, right_imgpoints,
                mtx_left, dist_left, mtx_right, dist_right,
                img_size,
                flags=flags,
                criteria=criteria
            )
        else:
            # Для OpenCV 3.x - возвращает 9 значений (но с другими именами)
            ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
                objpoints, left_imgpoints, right_imgpoints,
                mtx_left, dist_left, mtx_right, dist_right,
                img_size,
                flags=flags,
                criteria=criteria
            )
        
        print(f"\nСтереокалибровка завершена успешно!")
        print(f"RMS ошибка репроекции: {ret:.4f}")
        print(f"Матрица поворота R:\n{R}")
        print(f"Вектор трансляции T:\n{T}")
        
        # Проверка качества калибровки по ошибке репроекции
        if ret > 1.0:
            print(f"ВНИМАНИЕ: Высокая ошибка репроекции ({ret:.4f}). Качество калибровки может быть низким.")
        else:
            print("Ошибка репроекции в допустимых пределах.")
        
        # Вычисление базиса (расстояния между центрами камер)
        baseline_cm = np.linalg.norm(T)  # Евклидова норма вектора трансляции
        print(f"Базис (расстояние между камерами): {baseline_cm:.2f} {'см' if square_size == 2.65 else 'ед.'}")
        
        # Проверка реалистичности значения базиса
        if baseline_cm < 1.0 or baseline_cm > 50.0:
            print(f"ВНИМАНИЕ: Подозрительное значение базиса: {baseline_cm:.2f}")
        
        # ШАГ 8: Стереоректификация - выравнивание изображений для упрощения сопоставления
        print("Выполнение стереоректификации...")
        # stereoRectify вычисляет матрицы ректификации для выравнивания изображений
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx_left, dist_left,
            mtx_right, dist_right,
            img_size, R, T,
            alpha=0  # 0 - обрезать черные области, 1 - оставить все (могут быть черные поля)
        )
        
        # ШАГ 9: Сохранение результатов калибровки
        stereo_data = {
            'ret': ret,          # RMS ошибка репроекции
            'R': R,              # Матрица вращения (3x3) - поворот правой камеры относительно левой
            'T': T,              # Вектор трансляции (3x1) - смещение правой камеры относительно левой
            'E': E,              # Существенная матрица (Essential matrix)
            'F': F,              # Фундаментальная матрица (Fundamental matrix)
            'R1': R1,            # Матрица ректификации для левой камеры
            'R2': R2,            # Матрица ректификации для правой камеры
            'P1': P1,            # Матрица проекции для левой камеры после ректификации
            'P2': P2,            # Матрица проекции для правой камеры после ректификации
            'Q': Q,              # Матрица репроекции (disparity-to-depth)
            'roi1': roi1,        # Область интереса для левого изображения после ректификации
            'roi2': roi2,        # Область интереса для правого изображения после ректификации
            'mtx_left': mtx_left,        # Матрица камеры левой камеры
            'dist_left': dist_left,      # Коэффициенты дисторсии левой камеры
            'mtx_right': mtx_right,      # Матрица камеры правой камеры
            'dist_right': dist_right,    # Коэффициенты дисторсии правой камеры
            'chessboard_size': chessboard_size,  # Размер шахматной доски
            'square_size': square_size,          # Размер квадрата
            'img_size': img_size,                # Размер изображений
            'num_valid_pairs': found_count       # Количество использованных пар
        }
        
        # Сохраняем все данные в pickle-файл для последующего использования
        stereo_calib_file = os.path.join(output_dir, 'stereo_calibration_data.pkl')
        with open(stereo_calib_file, 'wb') as f:
            pickle.dump(stereo_data, f)
        
        # Дополнительно сохраняем ключевые матрицы в текстовые файлы для удобства
        np.savetxt(os.path.join(output_dir, 'rotation_matrix.txt'), R)      # Матрица вращения
        np.savetxt(os.path.join(output_dir, 'translation_vector.txt'), T)   # Вектор трансляции
        np.savetxt(os.path.join(output_dir, 'essential_matrix.txt'), E)     # Существенная матрица
        np.savetxt(os.path.join(output_dir, 'fundamental_matrix.txt'), F)   # Фундаментальная матрица
        np.savetxt(os.path.join(output_dir, 'Q_matrix.txt'), Q)             # Матрица репроекции
        
        print(f"\nРезультаты стереокалибровки сохранены в {output_dir}")
        print(f"Базис: {baseline_cm:.2f} {'см' if square_size == 2.65 else 'ед.'}")
        print(f"Размер ROI левой камеры: {roi1}")
        print(f"Размер ROI правой камеры: {roi2}")
        
        # ШАГ 10: Создание тестового изображения для проверки ректификации
        print("\nСоздание тестового изображения ректификации...")
        # Берем первую успешную стереопару для теста
        test_pair = valid_pairs[0]
        l_img = cv2.imread(test_pair[0])
        r_img = cv2.imread(test_pair[1])
        
        # Создаем карты ректификации для преобразования изображений
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, R1, P1, img_size, cv2.CV_16SC2
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            mtx_right, dist_right, R2, P2, img_size, cv2.CV_16SC2
        )
        
        # Применяем ректификацию к изображениям
        left_rectified = cv2.remap(l_img, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(r_img, right_map1, right_map2, cv2.INTER_LINEAR)
        
        # Соединяем левое и правое изображения горизонтально для визуального сравнения
        test_result = np.hstack((left_rectified, right_rectified))
        cv2.imwrite(os.path.join(output_dir, 'stereo_rectification_test.jpg'), test_result)
        print("Тестовое изображение ректификации сохранено как 'stereo_rectification_test.jpg'")
        
        print("\nПроцесс завершен успешно!")
        
    except Exception as e:
        print(f"Ошибка при выполнении стереокалибровки: {e}")
        import traceback
        traceback.print_exc()
        return

# Точка входа при запуске скрипта напрямую
if __name__ == "__main__":
    # Параметры по умолчанию для запуска напрямую
    calibrate(
        images_dir='captures_stereo',      # Директория с изображениями стереопар
        left_pattern='left_*.jpg',         # Шаблон для левых изображений
        right_pattern='right_*.jpg',       # Шаблон для правых изображений
        output_dir='output',               # Директория для сохранения результатов
        chessboard_size=(7, 7),            # Размер шахматной доски (7x7 углов)
        square_size=2.65                   # Размер квадрата в сантиметрах
    )