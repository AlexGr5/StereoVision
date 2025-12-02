import numpy as np
import cv2
import glob
import os
import pickle

# Параметры калибровки камеры
# Эти переменные можно изменять по необходимости
CHESSBOARD_SIZE = (7, 7)                              # Количество пересечений шахматной доски (ширина x высота)
SQUARE_SIZE = 2.65                                    # Реальный размер квадрата шахматной доски в сантиметрах
CALIBRATION_IMAGES_PATH = 'calibration_images/*.jpg'  # Путь к изображениям для калибровки
OUTPUT_DIRECTORY = 'output'                           # Директория для сохранения результатов
SAVE_UNDISTORTED = True                               # Сохранять ли исправленные (без дисторсии) изображения?
CALIBRATE_FILE_NAME = 'calibration_data.pkl'          # Имя файла для сохранения данных калибровки
LEFT_OR_RIGHT = ""                                    # Метка для левой/правой камеры (используется в invoke())

def calibrate_camera():
    """
    Калибрует камеру с использованием изображений шахматной доски.
    
    Возвращает:
        ret: Среднеквадратичная ошибка повторной проекции (RMS)
        mtx: Матрица камеры (внутренние параметры)
        dist: Коэффициенты дисторсии
        rvecs: Векторы вращения для каждого изображения
        tvecs: Векторы смещения для каждого изображения
    
    Процесс:
        1. Создает 3D точки реальной шахматной доски
        2. Находит углы шахматной доски на каждом изображении
        3. Использует соответствие 2D-3D точек для вычисления параметров камеры
    """
    # Подготовка массива 3D точек: (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    
    # Масштабирование 3D точек по реальному размеру квадрата (для измерений в реальном мире)
    objp = objp * SQUARE_SIZE
    
    # Массивы для хранения 3D точек (реальный мир) и 2D точек (изображение) со всех изображений
    objpoints = []  # 3D точки в реальном мире
    imgpoints = []  # 2D точки на плоскости изображения
    
    # Получение списка файлов изображений для калибровки
    images = glob.glob(CALIBRATION_IMAGES_PATH)
    
    if not images:
        print(f"Изображения для калибровки не найдены по пути: {CALIBRATION_IMAGES_PATH}")
        return None, None, None, None, None
    
    # Создание выходной директории, если она не существует
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    print(f"Найдено {len(images)} изображений для калибровки")
    
    # Обработка каждого калибровочного изображения
    for idx, fname in enumerate(images):
        # Загрузка изображения
        img = cv2.imread(fname)
        # Конвертация в оттенки серого для поиска углов
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Поиск углов шахматной доски на изображении
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        # Если углы найдены, добавляем точки в массивы
        if ret:
            objpoints.append(objp)  # Добавляем 3D точки (одинаковые для всех изображений)
            
            # Уточнение позиций углов с субпиксельной точностью
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Рисуем и отображаем найденные углы (для визуальной проверки)
            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
            
            # Сохраняем изображение с отмеченными углами
            output_img_path = os.path.join(OUTPUT_DIRECTORY, f'corners_{os.path.basename(fname)}')
            cv2.imwrite(output_img_path, img)
            
            print(f"Обработано изображение {idx+1}/{len(images)}: {fname} - Шахматная доска найдена")
        else:
            print(f"Обработано изображение {idx+1}/{len(images)}: {fname} - Шахматная доска НЕ найдена")
    
    # Проверка, найдены ли хотя бы одни углы на каком-либо изображении
    if not objpoints:
        print("Шахматная доска не была обнаружена ни на одном изображении.")
        return None, None, None, None, None
    
    print("Выполняется калибровка камеры...")
    
    # Калибровка камеры с использованием всех найденных точек
    # cv2.calibrateCamera находит матрицу камеры и коэффициенты дисторсии
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Сохранение результатов калибровки в словарь
    calibration_data = {
        'camera_matrix': mtx,               # Матрица камеры (3x3)
        'distortion_coefficients': dist,    # Коэффициенты дисторсии (обычно 5 параметров)
        'rotation_vectors': rvecs,          # Векторы вращения для каждого изображения
        'translation_vectors': tvecs,       # Векторы смещения для каждого изображения
        'reprojection_error': ret           # Общая ошибка повторной проекции
    }
    
    # Сохранение данных калибровки в файл pickle для последующего использования
    with open(os.path.join(OUTPUT_DIRECTORY, CALIBRATE_FILE_NAME), 'wb') as f:
        pickle.dump(calibration_data, f)
    
    # Дополнительное сохранение матрицы камеры и коэффициентов дисторсии в текстовых файлах
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'camera_matrix_' + LEFT_OR_RIGHT + '.txt'), mtx)
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'distortion_coefficients' + LEFT_OR_RIGHT + '.txt'), dist)
    
    print(f"Калибровка завершена! Среднеквадратичная ошибка повторной проекции: {ret}")
    print(f"Результаты сохранены в директории: {OUTPUT_DIRECTORY}")
    
    return ret, mtx, dist, rvecs, tvecs

def undistort_images(mtx, dist):
    """
    Устраняет дисторсию (искажение) на всех калибровочных изображениях.
    
    Параметры:
        mtx: Матрица камеры, полученная при калибровке
        dist: Коэффициенты дисторсии, полученные при калибровке
    
    Процесс:
        1. Для каждого изображения вычисляет оптимальную новую матрицу камеры
        2. Применяет преобразование для устранения дисторсии
        3. Обрезает черные области по краям (опционально)
        4. Сохраняет исправленные изображения
    """
    # Проверка флага сохранения исправленных изображений
    if not SAVE_UNDISTORTED:
        return
    
    # Получение списка изображений для обработки
    images = glob.glob(CALIBRATION_IMAGES_PATH)
    
    if not images:
        print(f"Изображения не найдены по пути: {CALIBRATION_IMAGES_PATH}")
        return
    
    # Создание директории для исправленных изображений
    undistorted_dir = os.path.join(OUTPUT_DIRECTORY, 'undistorted')
    if not os.path.exists(undistorted_dir):
        os.makedirs(undistorted_dir)
    
    print(f"Исправление дисторсии для {len(images)} изображений...")
    
    # Обработка каждого изображения
    for idx, fname in enumerate(images):
        # Загрузка изображения
        img = cv2.imread(fname)
        # Получение размеров изображения
        h, w = img.shape[:2]
        
        # Вычисление оптимальной новой матрицы камеры
        # Параметр alpha=1 сохраняет все пиксели (могут быть черные области)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # Устранение дисторсии на изображении
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # Обрезка изображения по ROI (область интереса) для удаления черных областей
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        # Сохранение исправленного изображения
        output_img_path = os.path.join(undistorted_dir, f'undistorted_{os.path.basename(fname)}')
        cv2.imwrite(output_img_path, dst)
        
        print(f"Исправлено изображение {idx+1}/{len(images)}: {fname}")
    
    print(f"Исправленные изображения сохранены в: {undistorted_dir}")

def calculate_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    """
    Вычисляет ошибку повторной проекции для каждого калибровочного изображения.
    
    Параметры:
        objpoints: Массив 3D точек реального мира для каждого изображения
        imgpoints: Массив 2D точек на изображении для каждого изображения
        mtx: Матрица камеры
        dist: Коэффициенты дисторсии
        rvecs: Векторы вращения для каждого изображения
        tvecs: Векторы смещения для каждого изображения
    
    Возвращает:
        mean_error: Средняя ошибка повторной проекции по всем изображениям
    
    Процесс:
        1. Для каждого изображения проецирует 3D точки обратно на 2D плоскость
        2. Сравнивает спроецированные точки с исходными найденными точками
        3. Вычисляет среднеквадратичную ошибку
    """
    total_error = 0
    
    # Вычисление ошибки для каждого изображения
    for i in range(len(objpoints)):
        # Проецирование 3D точек обратно на 2D плоскость с использованием параметров камеры
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        
        # Вычисление ошибки между исходными и спроецированными точками
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        print(f"Ошибка повторной проекции для изображения {i+1}: {error}")
    
    # Вычисление средней ошибки
    mean_error = total_error / len(objpoints)
    print(f"Средняя ошибка повторной проекции: {mean_error}")
    
    return mean_error

def main():
    """
    Основная функция, запускающая процесс калибровки камеры.
    
    Последовательность действий:
        1. Калибровка камеры с использованием изображений шахматной доски
        2. Исправление дисторсии на всех изображениях (если включено)
        3. Вывод информации о качестве калибровки
    """
    print("Запуск процесса калибровки камеры...")
    
    # Шаг 1: Калибровка камеры
    ret, mtx, dist, rvecs, tvecs = calibrate_camera()
    
    # Проверка успешности калибровки
    if mtx is None:
        print("Калибровка не удалась. Завершение работы.")
        return
    
    # Шаг 2: Исправление дисторсии на изображениях
    undistort_images(mtx, dist)
    
    print("Калибровка камеры успешно завершена!")

def invoke(input_dir, name_files, output_dir, left_or_right, size_chessboard, size_square):
    """
    Функция для вызова калибровки из внешних скриптов.
    
    Параметры:
        input_dir (str): Директория с изображениями для калибровки
        name_files (str): Шаблон имен файлов (например, 'left_*.jpg')
        output_dir (str): Директория для сохранения результатов
        left_or_right (str): Метка камеры ('left' или 'right')
        size_chessboard (tuple): Размер шахматной доски (ширина, высота)
        size_square (float): Реальный размер квадрата в сантиметрах
    
    Действия:
        Устанавливает глобальные параметры калибровки и запускает процесс калибровки.
    """
    # Установка глобальных переменных калибровки
    global CHESSBOARD_SIZE, SQUARE_SIZE, CALIBRATION_IMAGES_PATH, OUTPUT_DIRECTORY, SAVE_UNDISTORTED, CALIBRATE_FILE_NAME, LEFT_OR_RIGHT
    
    CHESSBOARD_SIZE = size_chessboard  # Размер шахматной доски
    SQUARE_SIZE = size_square  # Размер квадрата в сантиметрах
    CALIBRATION_IMAGES_PATH = input_dir + '/' + name_files  # Путь к изображениям
    OUTPUT_DIRECTORY = output_dir  # Выходная директория
    SAVE_UNDISTORTED = True  # Всегда сохранять исправленные изображения
    CALIBRATE_FILE_NAME = 'calibration_data' + '_' + left_or_right + '.pkl'  # Имя файла с учетом камеры
    LEFT_OR_RIGHT = left_or_right  # Метка камеры (left/right)
    
    # Запуск основного процесса калибровки
    main()

# Точка входа в программу
if __name__ == "__main__":
    main()