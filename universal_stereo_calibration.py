import numpy as np
import cv2
import glob
import os
import pickle
import re

def get_opencv_version():
    """
    Возвращает версию OpenCV в формате (major, minor, patch)
    
    Возвращает:
        tuple: Версия OpenCV в виде (мажорная, минорная, патч)
    
    Примечание:
        - Определяет версию OpenCV для обеспечения совместимости
        - В разных версиях OpenCV могут отличаться интерфейсы функций
    """
    version = cv2.__version__
    match = re.match(r'(\d+)\.(\d+)\.(\d+)', version)
    if match:
        return tuple(map(int, match.groups()))
    return (0, 0, 0)

def calibrate(images_dir, left_pattern, right_pattern, output_dir, chessboard_size, square_size):
    """
    Универсальная стереокалибровка - калибрует обе камеры и их взаимное расположение одновременно
    
    Параметры:
        images_dir (str): Директория с изображениями стереопар
        left_pattern (str): Шаблон имен файлов для левых изображений (например, 'left_*.jpg')
        right_pattern (str): Шаблон имен файлов для правых изображений (например, 'right_*.jpg')
        output_dir (str): Директория для сохранения результатов
        chessboard_size (tuple): Размер шахматной доски (ширина, высота) в углах
        square_size (float): Реальный размер квадрата шахматной доски в сантиметрах
    
    Основные отличия от обычной стереокалибровки:
        1. Не требует предварительной индивидуальной калибровки камер
        2. Одновременно определяет внутренние параметры обеих камер и их взаимное положение
        3. Используется, когда индивидуальная калибровка недоступна или ненадежна
    
    Процесс:
        1. Поиск и сопоставление стереопар изображений
        2. Обнаружение углов шахматной доски на каждой паре
        3. Выполнение универсальной стереокалибровки для одновременного определения
           всех параметров: внутренних параметров камер и их взаимного положения
        4. Стереоректификация для выравнивания изображений
        5. Сохранение результатов и создание тестового изображения
    """
    print("\n=== НАЧАЛО УНИВЕРСАЛЬНОЙ СТЕРЕОКАЛИБРОВКИ ===")
    print("Этот метод калибрует обе камеры и их взаимное расположение за один шаг")
    
    # ШАГ 1: Подготовка объектных точек (3D координаты углов шахматной доски в реальном мире)
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
    
    # ШАГ 2: Проверяем и сопоставляем пары изображений
    valid_pairs = []
    for l_img in left_images:
        base = os.path.basename(l_img).replace('left_', '')  # Убираем префикс 'left_'
        r_img = os.path.join(images_dir, 'right_' + base)    # Формируем имя правого файла
        if os.path.exists(r_img):
            valid_pairs.append((l_img, r_img))  # Добавляем пару, если оба файла существуют
    
    print(f"Найдено {len(valid_pairs)} валидных стереопар")
    
    # Требуется больше пар для универсальной калибровки (минимум 10)
    if len(valid_pairs) < 10:
        print(f"Ошибка: Недостаточно совпадающих пар изображений. Требуется минимум 10, найдено {len(valid_pairs)}")
        print("Убедитесь, что имена файлов соответствуют шаблонам left_*.jpg и right_*.jpg")
        return
    
    # Создаем директорию для сохранения изображений с отмеченными углами
    corners_dir = os.path.join(output_dir, 'universal_stereo_corners')
    os.makedirs(corners_dir, exist_ok=True)
    
    # ШАГ 3: Определяем размер изображений (предполагается, что все изображения одинакового размера)
    first_img = cv2.imread(valid_pairs[0][0])
    if first_img is None:
        print("Ошибка: Не удалось загрузить первое изображение для определения размера")
        return
    
    img_size = first_img.shape[1], first_img.shape[0]  # (ширина, высота)
    print(f"Размер изображений: {img_size}")
    
    # Критерии для уточнения позиций углов с субпиксельной точностью
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # ШАГ 4: Поиск углов шахматной доски на каждой стереопаре
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
            
            # Поиск углов шахматной доски
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)
            
            # Если углы не найдены стандартным методом, пробуем с дополнительными флагами
            if not ret_l or not ret_r:
                # Используем адаптивный порог и нормализацию изображения для улучшения поиска
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
    
    # Проверка достаточного количества успешных пар для калибровки (минимум 10)
    if found_count < 10:
        print(f"Ошибка: Найдено недостаточно валидных пар ({found_count}). Требуется минимум 10.")
        return
    
    print(f"\nНачало универсальной стереокалибровки с {found_count} валидными парами...")
    
    try:
        # ШАГ 5: Подготовка начальных приближений для универсальной стереокалибровки
        # В отличие от обычной стереокалибровки, здесь внутренние параметры неизвестны
        print("Запуск cv2.stereoCalibrate...")
        
        # Инициализируем матрицы камер как единичные
        cameraMatrix1 = np.eye(3, dtype=np.float64)  # Матрица камеры для левой камеры
        cameraMatrix2 = np.eye(3, dtype=np.float64)  # Матрица камеры для правой камеры
        
        # Устанавливаем начальные значения фокусного расстояния и оптического центра
        # Эти значения будут уточнены в процессе калибровки
        cameraMatrix1[0, 0] = img_size[0]  # fx - фокусное расстояние по x
        cameraMatrix1[1, 1] = img_size[1]  # fy - фокусное расстояние по y
        cameraMatrix1[0, 2] = img_size[0] / 2  # cx - координата x оптического центра
        cameraMatrix1[1, 2] = img_size[1] / 2  # cy - координата y оптического центра
        
        # Аналогично для правой камеры (предполагаем одинаковые камеры)
        cameraMatrix2[0, 0] = img_size[0]
        cameraMatrix2[1, 1] = img_size[1]
        cameraMatrix2[0, 2] = img_size[0] / 2
        cameraMatrix2[1, 2] = img_size[1] / 2
        
        # Инициализируем коэффициенты дисторсии нулями (предполагаем отсутствие дисторсии)
        distCoeffs1 = np.zeros((5, 1), dtype=np.float64)  # Коэффициенты дисторсии левой камеры
        distCoeffs2 = np.zeros((5, 1), dtype=np.float64)  # Коэффициенты дисторсии правой камеры
        
        # Флаги для универсальной стереокалибровки:
        # - CALIB_FIX_ASPECT_RATIO: фиксирует соотношение fx/fy
        # - CALIB_SAME_FOCAL_LENGTH: предполагает одинаковое фокусное расстояние для обеих камер
        # - CALIB_ZERO_TANGENT_DIST: обнуляет тангенциальные коэффициенты дисторсии
        # - CALIB_FIX_K3, CALIB_FIX_K4, CALIB_FIX_K5: фиксирует радиальные коэффициенты дисторсии
        flags = (cv2.CALIB_FIX_ASPECT_RATIO |
                 cv2.CALIB_SAME_FOCAL_LENGTH |
                 cv2.CALIB_ZERO_TANGENT_DIST |
                 cv2.CALIB_FIX_K3 |
                 cv2.CALIB_FIX_K4 |
                 cv2.CALIB_FIX_K5)
        
        # Критерии завершения оптимизации
        criteria_calib = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        # Определение версии OpenCV для совместимости
        opencv_version = get_opencv_version()
        print(f"Версия OpenCV: {opencv_version}")
        
        # ШАГ 6: Выполнение универсальной стереокалибровки
        # cv2.stereoCalibrate одновременно определяет:
        # - Внутренние параметры обеих камер (матрицы камер и коэффициенты дисторсии)
        # - Внешние параметры (взаимное положение камер)
        if opencv_version[0] >= 4:
            # Для OpenCV 4.x
            ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
                objpoints, left_imgpoints, right_imgpoints,
                cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                img_size,
                flags=flags,
                criteria=criteria_calib
            )
        else:
            # Для OpenCV 3.x
            ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
                objpoints, left_imgpoints, right_imgpoints,
                cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                img_size,
                flags=flags,
                criteria=criteria_calib
            )
        
        print(f"\nУниверсальная стереокалибровка завершена успешно!")
        print(f"RMS ошибка репроекции: {ret:.4f}")
        
        # Проверка качества калибровки по ошибке репроекции
        if ret > 1.0:
            print(f"ВНИМАНИЕ: Высокая ошибка репроекции ({ret:.4f}). Качество калибровки может быть низким.")
        else:
            print("Ошибка репроекции в допустимых пределах.")
        
        # Вычисление базиса (расстояния между центрами камер)
        baseline_cm = np.linalg.norm(T)  # Евклидова норма вектора трансляции
        print(f"Базис (расстояние между камерами): {baseline_cm:.2f} {'см' if square_size == 2.65 else 'ед.'}")
        
        # Проверка реалистичности значения базиса (обычно от 1 до 50 см для стереокамер)
        if baseline_cm < 1.0 or baseline_cm > 50.0:
            print(f"ВНИМАНИЕ: Подозрительное значение базиса: {baseline_cm:.2f}")
        
        # ШАГ 7: Стереоректификация - выравнивание изображений для упрощения сопоставления
        print("Выполнение стереоректификации...")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, img_size, R, T, alpha=0  # alpha=0 - обрезать черные области
        )
        
        # ШАГ 8: Сохранение результатов калибровки
        # Формат данных совместим с обычной стереокалибровкой
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
            'mtx_left': K1,      # Матрица камеры левой камеры (рассчитанная)
            'dist_left': D1,     # Коэффициенты дисторсии левой камеры (рассчитанные)
            'mtx_right': K2,     # Матрица камеры правой камеры (рассчитанная)
            'dist_right': D2,    # Коэффициенты дисторсии правой камеры (рассчитанные)
            'chessboard_size': chessboard_size,  # Размер шахматной доски
            'square_size': square_size,          # Размер квадрата
            'img_size': img_size,                # Размер изображений
            'num_valid_pairs': found_count,      # Количество использованных пар
            'calibration_type': 'universal_stereo'  # Помечаем тип калибровки для идентификации
        }
        
        # Сохраняем все данные в pickle-файл (тот же файл, что и при обычной стереокалибровке)
        stereo_calib_file = os.path.join(output_dir, 'stereo_calibration_data.pkl')
        with open(stereo_calib_file, 'wb') as f:
            pickle.dump(stereo_data, f)
        
        # Дополнительно сохраняем ключевые матрицы в текстовые файлы для удобства
        np.savetxt(os.path.join(output_dir, 'rotation_matrix.txt'), R)      # Матрица вращения
        np.savetxt(os.path.join(output_dir, 'translation_vector.txt'), T)   # Вектор трансляции
        np.savetxt(os.path.join(output_dir, 'essential_matrix.txt'), E)     # Существенная матрица
        np.savetxt(os.path.join(output_dir, 'fundamental_matrix.txt'), F)   # Фундаментальная матрица
        np.savetxt(os.path.join(output_dir, 'Q_matrix.txt'), Q)             # Матрица репроекции
        np.savetxt(os.path.join(output_dir, 'camera_matrix_left.txt'), K1)  # Матрица левой камеры
        np.savetxt(os.path.join(output_dir, 'camera_matrix_right.txt'), K2) # Матрица правой камеры
        np.savetxt(os.path.join(output_dir, 'distortion_left.txt'), D1)     # Дисторсия левой камеры
        np.savetxt(os.path.join(output_dir, 'distortion_right.txt'), D2)    # Дисторсия правой камеры
        
        print(f"\nРезультаты универсальной стереокалибровки сохранены в {output_dir}")
        print(f"Базис: {baseline_cm:.2f} {'см' if square_size == 2.65 else 'ед.'}")
        print(f"Размер ROI левой камеры: {roi1}")
        print(f"Размер ROI правой камеры: {roi2}")
        
        # Вывод детальной информации о результатах калибровки
        print(f"\nМатрица левой камеры:")
        print(K1)
        print(f"\nКоэффициенты дисторсии левой камеры:")
        print(D1)
        print(f"\nМатрица правой камеры:")
        print(K2)
        print(f"\nКоэффициенты дисторсии правой камеры:")
        print(D2)
        print(f"\nМатрица поворота между камерами:")
        print(R)
        print(f"\nВектор трансляции между камерами:")
        print(T)
        
        # ШАГ 9: Создание тестового изображения для проверки ректификации
        print("\nСоздание тестового изображения ректификации...")
        # Берем первую успешную стереопару для теста
        test_pair = valid_pairs[0]
        l_img = cv2.imread(test_pair[0])
        r_img = cv2.imread(test_pair[1])
        
        # Создаем карты ректификации для преобразования изображений
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, img_size, cv2.CV_16SC2
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2, img_size, cv2.CV_16SC2
        )
        
        # Применяем ректификацию к изображениям
        left_rectified = cv2.remap(l_img, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(r_img, right_map1, right_map2, cv2.INTER_LINEAR)
        
        # Соединяем левое и правое изображения горизонтально для визуального сравнения
        test_result = np.hstack((left_rectified, right_rectified))
        
        # Рисуем горизонтальные линии для проверки выравнивания
        # Если ректификация выполнена правильно, линии должны быть прямыми через оба изображения
        h, w = test_result.shape[:2]
        for y in range(0, h, 50):
            cv2.line(test_result, (0, y), (w, y), (0, 255, 0), 1)  # Зеленые линии через каждые 50 пикселей
        
        cv2.imwrite(os.path.join(output_dir, 'universal_stereo_rectification_test.jpg'), test_result)
        print("Тестовое изображение ректификации сохранено как 'universal_stereo_rectification_test.jpg'")
        
        print("\nУНИВЕРСАЛЬНАЯ СТЕРЕОКАЛИБРОВКА ЗАВЕРШЕНА УСПЕШНО!")
        print("Теперь можно использовать программу построения карты глубины!")
        
    except Exception as e:
        print(f"Ошибка при выполнении универсальной стереокалибровки: {e}")
        import traceback
        traceback.print_exc()
        return

# Точка входа при запуске скрипта напрямую
if __name__ == "__main__":
    # Параметры по умолчанию для запуска напрямую
    calibrate(
        images_dir='captures',      # Директория с изображениями стереопар
        left_pattern='left_*.jpg',  # Шаблон для левых изображений
        right_pattern='right_*.jpg',# Шаблон для правых изображений
        output_dir='output',        # Директория для сохранения результатов
        chessboard_size=(7, 7),     # Размер шахматной доски (7x7 углов)
        square_size=2.65            # Размер квадрата в сантиметрах
    )