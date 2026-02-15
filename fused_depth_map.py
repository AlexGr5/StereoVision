# -*- coding: utf-8 -*-
"""
СИСТЕМА ФЬЮЖНА КАРТ ГЛУБИНЫ С ОБНАРУЖЕНИЕМ ПЕРЕКРЫТИЯ КАМЕР

Эта система объединяет три независимых метода оценки глубины для создания надежной и точной карты глубины:
1. Стереозрение (основной метод) — вычисляет disparity map на основе пары откалиброванных камер
2. MiDaS (нейросетевая оценка) — резервный метод для заполнения областей с низкой достоверностью стерео
3. Оптический поток (структурный анализ) — заполняет "дыры" при движении камеры через анализ остаточного движения

ОСОБЕННОСТИ СИСТЕМЫ:
- Адаптивное переключение между методами при обнаружении перекрытия одной из камер
- Динамическая калибровка диапазонов глубины между разными методами
- Гистерезисное подавление ложных срабатываний детектора перекрытия
- Параллельная обработка методов через ThreadPoolExecutor для минимизации задержек
- Поддержка изменения масштаба обработки в реальном времени для баланса скорости/точности

"""

# ============================================================================
# ИМПОРТ СТАНДАРТНЫХ БИБЛИОТЕК
# ============================================================================

import cv2                     # OpenCV — основная библиотека для обработки изображений и работы с камерами
import numpy as np             # NumPy — для математических операций с массивами и матрицами
import os                      # Работа с файловой системой (проверка существования файлов, создание директорий)
import pickle                  # Сериализация/десериализация данных калибровки камер
import time                    # Таймеры для измерения производительности и управления задержками
import sys                     # Доступ к системной информации (платформа, аргументы командной строки)
import threading               # Поддержка многопоточности (для фоновых операций)
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError  # Управление пулом потоков для параллельной обработки

# ============================================================================
# ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ СИСТЕМЫ — КОНФИГУРАЦИЯ ОБРАБОТКИ
# ============================================================================

# МАСШТАБИРОВАНИЕ ИЗОБРАЖЕНИЙ ДЛЯ УСКОРЕНИЯ ОБРАБОТКИ:
# Значение 0.33 означает обработку изображений в 1/3 от оригинального разрешения.
# Теоретическое ускорение: 1/(0.33^2) ≈ 9.2×
PROCESSING_SCALE = 0.33
print(f"[ИНФО] Обработка на масштабе {PROCESSING_SCALE:.2f}x (ожидаемое ускорение ~{1/PROCESSING_SCALE**2:.1f}×)")

# ============================================================================
# ПОПЫТКА ИМПОРТА TORCH ДЛЯ ПОДДЕРЖКИ MiDaS
# ============================================================================

try:
    # Попытка импорта PyTorch — необходим для работы нейросетевой модели MiDaS
    import torch
    TORCH_AVAILABLE = True  # Флаг доступности PyTorch
    print("[ИНФО] PyTorch доступен. MiDaS будет активирован.")
except ImportError:
    # Если PyTorch не установлен — отключаем MiDaS, но продолжаем работу со стерео и оптическим потоком
    TORCH_AVAILABLE = False
    print("[ВНИМАНИЕ] PyTorch не установлен. MiDaS отключен.")

# ============================================================================
# КОНФИГУРАЦИОННЫЕ ПАРАМЕТРЫ СТЕРЕОЗРЕНИЯ
# ============================================================================

# Путь к файлу с данными калибровки стереопары (сгенерированному предварительно)
STEREO_CALIBRATION_FILE = 'output/stereo_calibration_data.pkl'

# СПИСОК ЖЕЛАЕМЫХ РАЗРЕШЕНИЙ ДЛЯ ПОИСКА КАМЕР:
# Система попытается установить одно из этих разрешений(слева направо) при инициализации камер
DESIRED_RESOLUTIONS = [(1920, 1080), (1280, 720), (640, 480)]

# РАЗРЕШЕНИЕ ДЛЯ ОТОБРАЖЕНИЯ В ОКНАХ:
# Все изображения для визуализации масштабируются до этого размера для удобства просмотра
DISPLAY_SIZE = (640, 480)

# БАЗОВЫЕ ПАРАМЕТРЫ АЛГОРИТМА StereoSGBM:
MIN_DISP_BASE = 0              # Минимальная диспаритет (смещение в пикселях)
NUM_DISP_BASE = 16 * 20        # Максимальная диспаритет (должна быть кратна 16 для OpenCV)
WINDOW_SIZE_BASE = 7           # Размер блока для сравнения (нечетное число, влияет на шум/детализацию)

# ============================================================================
# ПАРАМЕТРЫ ФЬЮЖНА (ОБЪЕДИНЕНИЯ) КАРТ ГЛУБИНЫ
# ============================================================================

# ВЕСА ВКЛАДА КАЖДОГО МЕТОДА В ИТОГОВУЮ КАРТУ:
FUSION_WEIGHTS = {
    'stereo_base': 0.8,        # Базовый вес стереозрения (основной метод)
    'midas_max_fill': 0.9,     # Максимальный вес заполнения областей MiDaS при низкой достоверности стерео
    'flow_max_fill': 0.5,      # Максимальный вес заполнения "дыр" оптическим потоком
}

# ПОРОГИ ДЛЯ ПРИНЯТИЯ РЕШЕНИЙ О ЗАПОЛНЕНИИ:
FUSION_THRESHOLDS = {
    'stereo_low_conf': 0.5,    # Порог достоверности стерео: ниже — заполнять резервными методами
    'midas_fill_start': 0.3,   # Порог начала заполнения MiDaS (не используется напрямую в коде)
    'flow_hole_threshold': 15, # Порог глубины для определения "дыры" (значения ниже — заполнять потоком)
    'flow_min_inliers': 15,    # Минимальное число инлайеров для валидации движения камеры через гомографию
}

# ПАРАМЕТРЫ СГЛАЖИВАНИЯ И БИЛАТЕРАЛЬНОЙ ФИЛЬТРАЦИИ:
FUSION_SMOOTHING = {
    'midas_blend_radius': 15,       # Радиус Гауссова размытия при смешивании MiDaS с основной картой
    'fusion_bilateral_d': 9,        # Диаметр окна билиатерального фильтра для финального сглаживания
    'fusion_bilateral_sigma': 75,   # Сигма для цветового и пространственного весов в билиатеральном фильтре
}

# ФЛАГИ АКТИВАЦИИ МЕТОДОВ (МОГУТ БЫТЬ ОТКЛЮЧЕНЫ ПОЛЬЗОВАТЕЛЕМ В РЕАЛЬНОМ ВРЕМЕНИ):
FUSION_METHODS = {
    'use_stereo': True,        # Использовать стереозрение как основной метод
    'use_midas_fill': True,    # Разрешить заполнение ненадежных областей MiDaS
    'use_flow_fill': True,     # Разрешить заполнение "дыр" оптическим потоком
}

# ТЕКУЩИЕ ПАРАМЕТРЫ ФЬЮЖНА (МОДИФИЦИРУЮТСЯ ПОЛЬЗОВАТЕЛЕМ ЧЕРЕЗ КЛАВИАТУРУ):
fusion_params = {
    'stereo_weight': FUSION_WEIGHTS['stereo_base'],        # Текущий вес стерео (0.1–1.5)
    'midas_fill_weight': FUSION_WEIGHTS['midas_max_fill'], # Текущий вес заполнения MiDaS (0.0–1.0)
    'flow_fill_weight': FUSION_WEIGHTS['flow_max_fill'],   # Текущий вес заполнения потоком (0.0–1.0)
    'stereo_conf_threshold': FUSION_THRESHOLDS['stereo_low_conf'],  # Порог достоверности стерео
    'flow_hole_threshold': FUSION_THRESHOLDS['flow_hole_threshold'], # Порог "дыр" для потока
}

# ВЫВОД ИНФОРМАЦИИ О ПАРАМЕТРАХ ФЬЮЖНА ДЛЯ ПОЛЬЗОВАТЕЛЯ:
print("=" * 70)
print("ПАРАМЕТРЫ ФЬЮЖНА (можно изменять в реальном времени):")
print(f"  Стерео базовый вес:        {fusion_params['stereo_weight']:.2f} (клавиши: W/S)")
print(f"  MiDaS макс. вес заполнения:{fusion_params['midas_fill_weight']:.2f} (клавиши: E/D)")
print(f"  Flow макс. вес заполнения: {fusion_params['flow_fill_weight']:.2f} (клавиши: R/F)")
print(f"  Порог уверенности стерео:  {fusion_params['stereo_conf_threshold']:.2f} (клавиши: T/G)")
print(f"  Порог 'дыр' для потока:    {fusion_params['flow_hole_threshold']:.1f} (клавиши: Y/H)")
print("=" * 70)

# ============================================================================
# ФУНКЦИЯ ОБНАРУЖЕНИЯ ПЕРЕКРЫТИЯ КАМЕР (ОККЛЮЗИИ)
# ============================================================================
def detect_camera_occlusion(left_img, right_img, occlusion_threshold=0.45):
    """
    Анализирует изображения с двух камер для обнаружения перекрытия (окклюзии) одной или обеих камер.
    
    АЛГОРИТМ РАБОТЫ:
    1. Вычисляет 5 метрик качества изображения для каждой камеры:
       - Стандартное отклонение яркости по блокам (гомогенность)
       - Доля блоков с низкой вариацией (текстура)
       - Контраст изображения (стандартное отклонение глобальное)
       - Энтропия (мера информативности)
       - Средняя яркость
    
    2. Сравнивает метрики между камерами и с абсолютными порогами:
       - Низкое стандартное отклонение → однородная поверхность (стена, потолок)
       - Высокая доля низковариативных блоков → отсутствие текстуры
       - Низкий контраст относительно другой камеры → возможное перекрытие
       - Низкая энтропия → отсутствие деталей
       - Сильное отличие яркости при низкой абсолютной яркости → закрытие объектива
    
    3. Формирует "оценку перекрытия" (0.0–1.0) для каждой камеры путем суммирования штрафов.
    
    4. Принимает решение по гистерезисному принципу:
       - 'left'  : левая камера перекрыта (оценка > порог, правая < 0.6*порог)
       - 'right' : правая камера перекрыта (аналогично)
       - 'both'  : обе камеры перекрыты (обе оценки > порог)
       - 'none'  : обе камеры свободны
    
    ПАРАМЕТРЫ:
    ----------
    left_img : np.ndarray
        Изображение с левой камеры (BGR или градации серого)
    right_img : np.ndarray
        Изображение с правой камеры (BGR или градации серого)
    occlusion_threshold : float, optional (default=0.45)
        Порог для принятия решения о перекрытии (0.0–1.0)
    
    ВОЗВРАЩАЕТ:
    ----------
    tuple : (result, left_score, right_score)
        result : str — 'none', 'left', 'right', 'both'
        left_score : float — оценка перекрытия левой камеры (0.0–1.0+)
        right_score : float — оценка перекрытия правой камеры (0.0–1.0+)
    """
    
    # Преобразование в градации серого, если изображение цветное (3 канала)
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if len(left_img.shape) == 3 else left_img
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if len(right_img.shape) == 3 else right_img

    # ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: вычисление гомогенности изображения по блокам
    def compute_block_homogeneity(gray, block_size=48):
        """
        Анализирует изображение блоками для оценки текстуры и однородности.
        
        ВОЗВРАЩАЕТ:
        ----------
        avg_std : float
            Среднее стандартное отклонение яркости по всем блокам
        low_var_ratio : float
            Доля блоков с очень низкой вариацией (стандартное отклонение < 12)
        """
        h, w = gray.shape  # Высота и ширина изображения
        # Количество блоков по вертикали и горизонтали
        blocks_h = max(1, h // block_size)
        blocks_w = max(1, w // block_size)
        std_values = []  # Список стандартных отклонений для каждого блока
        
        # Проход по всем блокам изображения
        for i in range(blocks_h):
            for j in range(blocks_w):
                # Координаты текущего блока (с обработкой краев изображения)
                y1 = i * block_size
                y2 = min((i + 1) * block_size, h)
                x1 = j * block_size
                x2 = min((j + 1) * block_size, w)
                block = gray[y1:y2, x1:x2]
                
                # Вычисление стандартного отклонения для блока, если он не пустой
                if block.size > 0:
                    std_values.append(np.std(block))
        
        # Среднее стандартное отклонение по всем блокам (0.0 если блоков нет)
        avg_std = np.mean(std_values) if std_values else 0
        # Доля блоков с низкой вариацией (менее 12 — почти однородные области)
        low_var_ratio = np.sum(np.array(std_values) < 12) / max(len(std_values), 1)
        return avg_std, low_var_ratio

    # Вычисление метрик гомогенности для обеих камер
    left_std, left_low_var = compute_block_homogeneity(gray_left)
    right_std, right_low_var = compute_block_homogeneity(gray_right)

    # ГЛОБАЛЬНЫЙ КОНТРАСТ: стандартное отклонение яркости по всему изображению
    left_contrast = np.std(gray_left)
    right_contrast = np.std(gray_right)

    # ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: вычисление энтропии изображения (мера информативности)
    def compute_entropy(gray):
        """
        Вычисляет энтропию изображения по гистограмме яркости.
        Высокая энтропия = много деталей/текстуры, низкая = однородная область.
        """
        # Построение гистограммы яркости (256 бинов для 8-битного изображения)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() + 1e-10  # Добавление эпсилон для избежания деления на ноль
        hist = hist / hist.sum()       # Нормализация гистограммы в вероятности
        # Формула Шеннона для энтропии
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy

    # Вычисление энтропии для обеих камер
    left_entropy = compute_entropy(gray_left)
    right_entropy = compute_entropy(gray_right)

    # СРЕДНЯЯ ЯРКОСТЬ ИЗОБРАЖЕНИЯ
    left_brightness = np.mean(gray_left)
    right_brightness = np.mean(gray_right)

    # АБСОЛЮТНЫЕ ПОРОГИ ДЛЯ МЕТРИК (ЭМПИРИЧЕСКИ ПОДОБРАНЫ):
    STD_THRESHOLD = 28.0          # Минимальное приемлемое стандартное отклонение по блокам
    LOW_VAR_THRESHOLD = 0.55      # Максимальная допустимая доля низковариативных блоков
    CONTRAST_RATIO = 2.2          # Отношение контрастов для сравнения камер
    ENTROPY_RATIO = 1.6           # Отношение энтропий для сравнения камер
    BRIGHTNESS_DIFF = 45.0        # Максимальная допустимая разница яркости между камерами

    # ИНИЦИАЛИЗАЦИЯ ОЦЕНОК ПЕРЕКРЫТИЯ (НАЧИНАЕМ С 0.0)
    left_occlusion_score = 0.0
    right_occlusion_score = 0.0

    # АНАЛИЗ ЛЕВОЙ КАМЕРЫ: накопление штрафов за признаки перекрытия
    if left_std < STD_THRESHOLD * 0.8:
        # Слишком низкое стандартное отклонение по блокам → однородная поверхность
        left_occlusion_score += 0.35
    if left_low_var > LOW_VAR_THRESHOLD:
        # Слишком много низковариативных блоков → отсутствие текстуры
        left_occlusion_score += 0.35
    if left_contrast < right_contrast / CONTRAST_RATIO and right_contrast > 15:
        # Контраст значительно ниже правой камеры (при условии, что правая имеет достаточный контраст)
        left_occlusion_score += 0.25
    if left_entropy < right_entropy / ENTROPY_RATIO and right_entropy > 5.0:
        # Энтропия значительно ниже правой камеры (при условии информативности правой)
        left_occlusion_score += 0.25
    if abs(left_brightness - right_brightness) > BRIGHTNESS_DIFF and left_brightness < 80:
        # Сильная разница яркости И низкая абсолютная яркость → возможное закрытие объектива
        left_occlusion_score += 0.2

    # АНАЛИЗ ПРАВОЙ КАМЕРЫ: симметричные проверки
    if right_std < STD_THRESHOLD * 0.8:
        right_occlusion_score += 0.35
    if right_low_var > LOW_VAR_THRESHOLD:
        right_occlusion_score += 0.35
    if right_contrast < left_contrast / CONTRAST_RATIO and left_contrast > 15:
        right_occlusion_score += 0.25
    if right_entropy < left_entropy / ENTROPY_RATIO and left_entropy > 5.0:
        right_occlusion_score += 0.25
    if abs(right_brightness - left_brightness) > BRIGHTNESS_DIFF and right_brightness < 80:
        right_occlusion_score += 0.2

    # ПРИНЯТИЕ РЕШЕНИЯ НА ОСНОВЕ ОЦЕНОК:
    if left_occlusion_score > occlusion_threshold and right_occlusion_score < occlusion_threshold * 0.6:
        # Левая перекрыта, правая свободна
        result = 'left'
    elif right_occlusion_score > occlusion_threshold and left_occlusion_score < occlusion_threshold * 0.6:
        # Правая перекрыта, левая свободна
        result = 'right'
    elif left_occlusion_score > occlusion_threshold and right_occlusion_score > occlusion_threshold:
        # Обе камеры перекрыты (редкий случай, стерео все равно работает хуже)
        result = 'both'
    else:
        # Обе камеры свободны — нормальная работа стерео
        result = 'none'

    return result, left_occlusion_score, right_occlusion_score

# ============================================================================
# ФУНКЦИИ ДЛЯ СТЕРЕОЗРЕНИЯ — ЗАГРУЗКА КАЛИБРОВКИ И РЕКТИФИКАЦИЯ
# ============================================================================

def load_stereo_calibration_with_scaling(scale_factor=1.0):
    """
    Загружает данные калибровки стереопары и генерирует карты ректификации с учетом масштаба обработки.
    
    РЕКТИФИКАЦИЯ — критически важный этап, который:
    1. Исправляет дисторсию линз (радиальная и тангенциальная)
    2. Выравнивает эпиполярные линии в горизонтальные прямые
    3. Приводит изображения к единой проекционной плоскости
    
    МАСШТАБИРОВАНИЕ ПАРАМЕТРОВ КАЛИБРОВКИ:
    При обработке в масштабе <1.0 необходимо скорректировать:
    - Фокусные расстояния (fx, fy) — умножаются на scale_factor
    - Главные точки (cx, cy) — умножаются на scale_factor
    - Размер изображения для ректификации — пропорционально уменьшается
    
    ПАРАМЕТРЫ:
    ----------
    scale_factor : float, optional (default=1.0)
        Коэффициент масштабирования (0.3–1.0). 1.0 = оригинальное разрешение.
    
    ВОЗВРАЩАЕТ:
    ----------
    stereo_calib : dict или None
        Словарь с параметрами ректификации или None при ошибке.
        Содержит:
        - Карты undistort/rectify (left_map1, left_map2, ...)
        - ROI для обрезки изображений после ректификации
        - Матрицу Q для преобразования disparity → 3D точка
        - Базис (расстояние между камерами)
        - Масштабированные матрицы камеры и дисторсии
        - Оригинальное и обработанное разрешения
    """
    
    # Проверка существования файла калибровки
    if not os.path.exists(STEREO_CALIBRATION_FILE):
        print(f"❌ Ошибка: файл стереокалибровки не найден: {STEREO_CALIBRATION_FILE}")
        return None
    
    try:
        # Загрузка сериализованных данных калибровки
        with open(STEREO_CALIBRATION_FILE, 'rb') as f:
            stereo_data = pickle.load(f)
        print("✅ Данные стереокалибровки успешно загружены")
        
        # Извлечение параметров калибровки из файла
        mtx_left_orig = stereo_data['mtx_left']    # Матрица левой камеры (оригинал)
        dist_left = stereo_data['dist_left']       # Коэффициенты дисторсии левой камеры
        mtx_right_orig = stereo_data['mtx_right']  # Матрица правой камеры (оригинал)
        dist_right = stereo_data['dist_right']     # Коэффициенты дисторсии правой камеры
        R = stereo_data['R']                       # Матрица поворота между камерами
        T = stereo_data['T']                       # Вектор трансляции между камерами
        img_size_orig = tuple(stereo_data['img_size'])  # Оригинальное разрешение калибровки
        
        print(f"Оригинальное калибровочное разрешение: {img_size_orig}")
        print(f"Масштаб обработки: {scale_factor:.2f}x")
        
        # МАСШТАБИРОВАНИЕ МАТРИЦ КАМЕР:
        # Копируем оригинальные матрицы, чтобы не изменять исходные данные
        mtx_left_scaled = mtx_left_orig.copy()
        mtx_right_scaled = mtx_right_orig.copy()
        
        # Коррекция фокусных расстояний (элементы [0,0] и [1,1])
        mtx_left_scaled[0, 0] *= scale_factor
        mtx_left_scaled[1, 1] *= scale_factor
        mtx_right_scaled[0, 0] *= scale_factor
        mtx_right_scaled[1, 1] *= scale_factor
        
        # Коррекция главных точек (элементы [0,2] и [1,2])
        mtx_left_scaled[0, 2] *= scale_factor
        mtx_left_scaled[1, 2] *= scale_factor
        mtx_right_scaled[0, 2] *= scale_factor
        mtx_right_scaled[1, 2] *= scale_factor
        
        print("[РЕКТЕФИКАЦИЯ] Масштабированные фокусные расстояния:")
        print(f"  Левая камера: fx={mtx_left_scaled[0,0]:.2f}, fy={mtx_left_scaled[1,1]:.2f}")
        print(f"  Правая камера: fx={mtx_right_scaled[0,0]:.2f}, fy={mtx_right_scaled[1,1]:.2f}")
        
        # ВЫЧИСЛЕНИЕ НОВОГО РАЗРЕШЕНИЯ ДЛЯ ОБРАБОТКИ:
        new_width = int(img_size_orig[0] * scale_factor)
        new_height = int(img_size_orig[1] * scale_factor)
        img_size_scaled = (new_width, new_height)
        print(f"[РЕКТЕФИКАЦИЯ] Новое разрешение для обработки: {img_size_scaled}")
        
        # ГЕНЕРАЦИЯ МАТРИЦ РЕКТИФИКАЦИИ ЧЕРЕЗ stereoRectify:
        # alpha=0 — обрезка до области с полным перекрытием (без черных полос)
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx_left_scaled, dist_left,
            mtx_right_scaled, dist_right,
            img_size_scaled,
            R, T,
            alpha=0,  # Полная обрезка до перекрывающейся области
            flags=cv2.CALIB_ZERO_DISPARITY  # Смещение disparity начинается с 0
        )
        
        # ГЕНЕРАЦИЯ КАРТ ДЛЯ ПЕРЕМЕЩЕНИЯ ПИКСЕЛЕЙ (remap):
        # left_map1 — координаты X для каждого пикселя после ректификации
        # left_map2 — координаты Y для каждого пикселя после ректификации
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            mtx_left_scaled, dist_left, R1, P1, img_size_scaled, cv2.CV_16SC2
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            mtx_right_scaled, dist_right, R2, P2, img_size_scaled, cv2.CV_16SC2
        )
        
        # ФОРМИРОВАНИЕ СЛОВАРЯ С ПАРАМЕТРАМИ РЕКТИФИКАЦИИ:
        stereo_calib = {
            'left_map1': left_map1,        # Карта X для левой камеры
            'left_map2': left_map2,        # Карта Y для левой камеры
            'right_map1': right_map1,      # Карта X для правой камеры
            'right_map2': right_map2,      # Карта Y для правой камеры
            'roi1': roi1,                  # ROI после ректификации (левая)
            'roi2': roi2,                  # ROI после ректификации (правая)
            'Q': Q,                        # Матрица для преобразования disparity → 3D
            'R': R,                        # Матрица поворота между камерами
            'T': T,                        # Вектор трансляции между камерами
            'mtx_left': mtx_left_scaled,   # Масштабированная матрица левой камеры
            'dist_left': dist_left,        # Коэффициенты дисторсии левой камеры
            'mtx_right': mtx_right_scaled, # Масштабированная матрица правой камеры
            'dist_right': dist_right,      # Коэффициенты дисторсии правой камеры
            'baseline': abs(T[0, 0]),      # Базис — расстояние между камерами в метрах
            'img_size_orig': img_size_orig,# Оригинальное разрешение калибровки
            'img_size_proc': img_size_scaled,  # Разрешение для обработки
            'focal_length': mtx_left_scaled[0, 0],  # Фокусное расстояние (пиксели)
            'scale_factor': scale_factor   # Текущий масштаб обработки
        }
        
        print(f"Стереоректификация подготовлена. Базис: {abs(T[0, 0]):.4f} м")
        print(f"[РЕКТЕФИКАЦИЯ] ✅ Карты успешно сгенерированы для разрешения {img_size_scaled}")
        return stereo_calib
        
    except Exception as e:
        # Обработка любых исключений при загрузке калибровки
        print(f"❌ Ошибка при загрузке стереокалибровки: {e}")
        import traceback
        traceback.print_exc()
        return None

def apply_stereo_rectification(left_img, right_img, stereo_calib):
    """
    Применяет ректификацию к паре изображений с использованием предварительно сгенерированных карт.
    
    ПАРАМЕТРЫ:
    ----------
    left_img : np.ndarray
        Исходное изображение с левой камеры
    right_img : np.ndarray
        Исходное изображение с правой камеры
    stereo_calib : dict
        Словарь с параметрами ректификации (результат load_stereo_calibration_with_scaling)
    
    ВОЗВРАЩАЕТ:
    ----------
    tuple : (left_rectified, right_rectified)
        Ректифицированные изображения (или исходные при ошибке)
    """
    
    # Проверка наличия данных калибровки
    if stereo_calib is None:
        return left_img, right_img
    
    try:
        # Получение целевого разрешения для обработки
        proc_w, proc_h = stereo_calib['img_size_proc']
        
        # МАСШТАБИРОВАНИЕ ИЗОБРАЖЕНИЙ ДО ЦЕЛЕВОГО РАЗРЕШЕНИЯ:
        # Необходимо, если камеры работают в другом разрешении, чем калибровка
        if (left_img.shape[1], left_img.shape[0]) != (proc_w, proc_h):
            left_img = cv2.resize(left_img, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        if (right_img.shape[1], right_img.shape[0]) != (proc_w, proc_h):
            right_img = cv2.resize(right_img, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        
        # ПРИМЕНЕНИЕ РЕКТИФИКАЦИИ ЧЕРЕЗ ПЕРЕМЕЩЕНИЕ ПИКСЕЛЕЙ (remap):
        # left_map1/left_map2 содержат координаты исходных пикселей для каждого пикселя результата
        left_rectified = cv2.remap(
            left_img,
            stereo_calib['left_map1'],
            stereo_calib['left_map2'],
            cv2.INTER_LINEAR  # Линейная интерполяция для сглаживания
        )
        right_rectified = cv2.remap(
            right_img,
            stereo_calib['right_map1'],
            stereo_calib['right_map2'],
            cv2.INTER_LINEAR
        )
        
        return left_rectified, right_rectified
        
    except Exception as e:
        # При ошибке возвращаем исходные изображения для продолжения работы
        print(f"Ошибка при стереоректификации: {e}")
        import traceback
        traceback.print_exc()
        return left_img, right_img

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ — РАБОТА С КАМЕРАМИ И ИЗОБРАЖЕНИЯМИ
# ============================================================================

def ensure_same_size(left_img, right_img):
    """
    Приводит два изображения к одинаковому размеру путем масштабирования до минимальных размеров.
    
    ПАРАМЕТРЫ:
    ----------
    left_img : np.ndarray
        Изображение с левой камеры
    right_img : np.ndarray
        Изображение с правой камеры
    
    ВОЗВРАЩАЕТ:
    ----------
    tuple : (left_resized, right_resized)
        Изображения одинакового размера
    """
    h1, w1 = left_img.shape[:2]  # Высота и ширина левого изображения
    h2, w2 = right_img.shape[:2]  # Высота и ширина правого изображения
    
    # Если размеры совпадают — возвращаем без изменений
    if (h1, w1) == (h2, w2):
        return left_img, right_img
    
    # Определение минимальных размеров для обрезки/масштабирования
    h_min = min(h1, h2)
    w_min = min(w1, w2)
    
    # Масштабирование обоих изображений до минимальных размеров
    left_resized = cv2.resize(left_img, (w_min, h_min))
    right_resized = cv2.resize(right_img, (w_min, h_min))
    
    return left_resized, right_resized

def find_available_cameras(max_test=10):
    """
    Поиск и тестирование доступных камер в системе с автоматическим определением разрешения.
    
    АЛГОРИТМ:
    1. Перебирает индексы камер от 0 до max_test-1
    2. Для каждой камеры пытается установить одно из желаемых разрешений
    3. Проверяет успешность установки через чтение кадра и анализ размера
    4. Сбрасывает буфер камеры перед тестированием для избежания старых кадров
    
    ПАРАМЕТРЫ:
    ----------
    max_test : int, optional (default=10)
        Максимальный индекс камеры для проверки
    
    ВОЗВРАЩАЕТ:
    ----------
    list : Список словарей с информацией о доступных камерах
        Каждый элемент содержит:
        - 'id': индекс камеры в системе
        - 'cap': объект VideoCapture (открытый!)
        - 'resolution': кортеж (ширина, высота) реального разрешения
        - 'fps': частота кадров
    """
    
    available_cameras = []  # Список найденных камер
    
    print("\n" + "=" * 60)
    print("ПОИСК И ПРОВЕРКА КАМЕР")
    print("=" * 60)
    print("\nИдет поиск камер...")
    
    # СПИСОК ЦЕЛЕВЫХ РАЗРЕШЕНИЙ ДЛЯ ТЕСТИРОВАНИЯ (от высокого к низкому)
    target_resolutions = [(1920, 1080), (1280, 720), (640, 480)]
    
    # Перебор индексов камер
    for i in range(max_test):
        # ЯВНОЕ УКАЗАНИЕ БЭКЕНДА ДЛЯ КРОССПЛАТФОРМЕННОСТИ:
        # macOS требует CAP_AVFOUNDATION, Windows — CAP_MSMF для стабильности
        if sys.platform == 'darwin':
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        
        # Пропуск если камера не открылась
        if not cap.isOpened():
            continue
        
        # ПОПЫТКА УСТАНОВКИ РАЗРЕШЕНИЯ:
        success = False
        for width, height in target_resolutions:
            # Установка желаемого разрешения
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            time.sleep(0.2)  # Пауза для применения настроек (критично для macOS)
            
            # Проверка реально установленного разрешения
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # СБРОС БУФЕРА КАМЕРЫ (5 кадров) для получения свежего изображения
            for _ in range(5):
                cap.read()
            
            # Чтение тестового кадра и проверка его валидности
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0 and actual_w >= width * 0.9:
                # Успешная инициализация камеры
                print(f"  Найдена камера {i}: {actual_w}x{actual_h} @ {cap.get(cv2.CAP_PROP_FPS):.1f} FPS")
                camera_info = {
                    'id': i,
                    'cap': cap,  # ВАЖНО: объект остается открытым!
                    'resolution': (actual_w, actual_h),
                    'fps': cap.get(cv2.CAP_PROP_FPS)
                }
                available_cameras.append(camera_info)
                success = True
                break  # Переход к следующей камере
        
        # Если ни одно разрешение не подошло — освобождаем ресурсы камеры
        if not success:
            cap.release()
    
    print(f"\nНайдено камер: {len(available_cameras)}")
    return available_cameras

def select_cameras_visual(available_cameras):
    """
    Визуальный интерфейс для выбора двух камер из списка доступных.
    
    ИНТЕРФЕЙС:
    - Окно с инструкциями и статусом выбора
    - Отдельные окна предпросмотра для каждой камеры
    - Визуальная индикация выбранных камер (рамки разного цвета)
    - Поддержка выбора через цифровые клавиши 0-9
    
    ПАРАМЕТРЫ:
    ----------
    available_cameras : list
        Список камер из find_available_cameras()
    
    ВОЗВРАЩАЕТ:
    ----------
    tuple : (left_id, right_id, left_cap, right_cap) или (None, None, None, None)
        Индексы и объекты выбранных камер или None при отмене
    """
    
    # Проверка минимального количества камер
    if len(available_cameras) < 2:
        print(f"Ошибка: найдено только {len(available_cameras)} камер. Нужно минимум 2.")
        return None, None, None, None
    
    print("\n" + "=" * 60)
    print("ВИЗУАЛЬНЫЙ ВЫБОР КАМЕР")
    print("=" * 60)
    print("\nИнструкции:")
    print("1. Нажмите цифру от 0 до 9, чтобы выбрать камеру с соответствующим ID")
    print("2. Выберите две камеры для стереопары")
    print("3. ESC - отмена выбора")
    
    selected_ids = []  # Список выбранных ID камер
    
    # СОЗДАНИЕ ИНСТРУКТИВНОГО ИЗОБРАЖЕНИЯ:
    instructions = np.zeros((200, 600, 3), dtype=np.uint8)  # Черный фон 600x200
    cv2.putText(instructions, "ВИЗУАЛЬНЫЙ ВЫБОР КАМЕР", (50, 40),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(instructions, "Нажмите цифру, чтобы выбрать камеру с таким ID", (50, 80),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(instructions, "ESC - отмена | ENTER - подтвердить выбор", (50, 110),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    
    # macOS FIX: установка пустого обработчика клика для предотвращения потери фокуса
    if sys.platform == 'darwin':
        cv2.namedWindow("Choice cameras", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Choice cameras", lambda *args: None)
    
    # ГЛАВНЫЙ ЦИКЛ ВЫБОРА:
    while True:
        # КОПИРОВАНИЕ ИНСТРУКЦИЙ ДЛЯ ОБНОВЛЕНИЯ СТАТУСА
        instructions_copy = instructions.copy()
        
        # ОТОБРАЖЕНИЕ ТЕКУЩЕГО СТАТУСА ВЫБОРА
        status = f"Выбранные камеры: {selected_ids}" if selected_ids else "Выбранные камеры: нет"
        cv2.putText(instructions_copy, status, (50, 140),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
        
        # ПОДСКАЗКА О КОЛИЧЕСТВЕ ОСТАВШИХСЯ ВЫБОРОВ
        if len(selected_ids) < 2:
            cv2.putText(instructions_copy, f"Выберите еще {2 - len(selected_ids)} камеру(ы)", (50, 170),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.putText(instructions_copy, "Нажмите ENTER для подтверждения", (50, 170),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        
        # ОТОБРАЖЕНИЕ ОКНА С ИНСТРУКЦИЯМИ
        cv2.imshow("Choice cameras", instructions_copy)
        
        # ОТОБРАЖЕНИЕ ПРЕДПРОСМОТРА ДЛЯ КАЖДОЙ ДОСТУПНОЙ КАМЕРЫ
        for cam in available_cameras:
            cap = cam['cap']
            camera_id = cam['id']
            ret, frame = cap.read()
            
            # ЗАЩИТА ОТ НЕКОРРЕКТНЫХ КАДРОВ (битые/пустые кадры)
            if not ret or frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                continue
            
            # МАСШТАБИРОВАНИЕ ДЛЯ ПРЕДПРОСМОТРА (320x240)
            display_frame = cv2.resize(frame, (320, 240))
            
            # ВИЗУАЛЬНАЯ ИНДИКАЦИЯ ВЫБРАННЫХ КАМЕР:
            if camera_id in selected_ids:
                idx = selected_ids.index(camera_id)
                color = (0, 255, 0) if idx == 0 else (255, 0, 0)  # Зеленая для первой, красная для второй
                cv2.rectangle(display_frame, (0, 0), (319, 239), color, 3)
                cv2.putText(display_frame, f"Выбрана #{idx+1}", (10, 220),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)
            
            # НАЛОЖЕНИЕ ИНФОРМАЦИИ О КАМЕРЕ
            cv2.putText(display_frame, f"ID: {camera_id}", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Нажмите {camera_id}", (10, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
            
            # macOS FIX: обработчик клика для окна предпросмотра
            if sys.platform == 'darwin':
                cv2.namedWindow(f"Preview camera {camera_id}", cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(f"Preview camera {camera_id}", lambda *args: None)
            
            # ОТОБРАЖЕНИЕ ОКНА ПРЕДПРОСМОТРА
            cv2.imshow(f"Preview camera {camera_id}", display_frame)
        
        # ОЖИДАНИЕ НАЖАТИЯ КЛАВИШИ (100 мс для плавности)
        key = cv2.waitKey(100) & 0xFF
        
        # ОБРАБОТКА КЛАВИШ:
        if key == 27:  # ESC — отмена выбора
            break
        elif key == 13 and len(selected_ids) == 2:  # ENTER — подтверждение выбора
            cv2.destroyAllWindows()
            
            # СОХРАНЕНИЕ ТОЛЬКО ВЫБРАННЫХ КАМЕР, ОСТАЛЬНЫЕ ЗАКРЫВАЕМ
            kept_caps = {}
            for cam in available_cameras:
                if cam['id'] in selected_ids:
                    kept_caps[cam['id']] = cam['cap']  # Сохраняем открытый объект
                else:
                    cam['cap'].release()  # Освобождаем неиспользуемые камеры
            
            # Возврат выбранных камер в порядке выбора
            return selected_ids[0], selected_ids[1], kept_caps[selected_ids[0]], kept_caps[selected_ids[1]]
        elif 48 <= key <= 57:  # Цифровые клавиши 0-9
            camera_id = key - 48
            available_ids = [cam['id'] for cam in available_cameras]
            # Добавление камеры в выбор если она доступна и еще не выбрана
            if camera_id in available_ids and camera_id not in selected_ids and len(selected_ids) < 2:
                selected_ids.append(camera_id)
    
    # ОЧИСТКА РЕСУРСОВ ПРИ ОТМЕНЕ
    cv2.destroyAllWindows()
    for cam in available_cameras:
        cam['cap'].release()
    return None, None, None, None

def determine_left_right_with_existing_caps(cam1_id, cam2_id, cap1, cap2):
    """
    Визуальное определение физического расположения камер (левая/правая) через интерактивный интерфейс.
    
    Индексы камер в системе (0, 1, 2...) не соответствуют их физическому расположению.
    Для корректной работы стерео необходимо знать, какая камера находится слева, какая — справа.
    
    ИНТЕРФЕЙС:
    - Горизонтальная компоновка двух изображений
    - Подсказки с клавишами управления
    - Возможность переключения назначения через 'L'/'R'
    
    ПАРАМЕТРЫ:
    ----------
    cam1_id, cam2_id : int
        Индексы двух выбранных камер
    cap1, cap2 : cv2.VideoCapture
        Открытые объекты камер
    
    ВОЗВРАЩАЕТ:
    ----------
    tuple : (left_id, right_id, left_cap, right_cap) или (None, None, None, None)
        Корректно назначенные левая и правая камеры
    """
    
    print(f"\nОпределение расположения камер {cam1_id} и {cam2_id}...")
    
    # Проверка состояния камер
    if not cap1.isOpened() or not cap2.isOpened():
        print("❌ Ошибка: камеры не открыты")
        return None, None, None, None
    
    # СБРОС БУФЕРА КАМЕР (3 кадра) для получения свежих изображений
    for _ in range(3):
        cap1.read()
        cap2.read()
    time.sleep(0.2)  # Дополнительная пауза для стабилизации
    
    print("\nИнструкции:")
    print("  'L' - если слева ЛЕВАЯ камера")
    print("  'R' - если слева ПРАВАЯ камера")
    print("  'Q' - отмена выбора")
    
    # macOS FIX: обработчик клика для окна определения
    if sys.platform == 'darwin':
        cv2.namedWindow('Detect left/right camera', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Detect left/right camera', lambda *args: None)
    
    frame_failures = 0  # Счетчик неудачных чтений кадров
    
    # ГЛАВНЫЙ ЦИКЛ ОПРЕДЕЛЕНИЯ:
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # ЗАЩИТА ОТ БИТЫХ КАДРОВ (многоуровневая проверка)
        if not ret1 or not ret2 or frame1 is None or frame2 is None or \
           frame1.size == 0 or frame2.size == 0 or \
           frame1.shape[0] == 0 or frame1.shape[1] == 0 or \
           frame2.shape[0] == 0 or frame2.shape[1] == 0:
            frame_failures += 1
            if frame_failures > 10:  # Превышение лимита неудач — выход
                print("❌ Слишком много неудачных кадров.")
                return None, None, None, None
            time.sleep(0.1)
            continue
        
        frame_failures = 0  # Сброс счетчика при успешном чтении
        
        # МАСШТАБИРОВАНИЕ ДЛЯ УДОБНОГО ПРЕДПРОСМОТРА (640x480)
        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))
        
        # СОЗДАНИЕ КОМБИНИРОВАННОГО ИЗОБРАЖЕНИЯ (горизонтальная стыковка)
        combined = np.hstack((frame1, frame2))
        
        # НАЛОЖЕНИЕ ИНФОРМАЦИИ О КАМЕРАХ
        cv2.putText(combined, f"Камера {cam1_id}", (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, f"Камера {cam2_id}", (650, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Определите ЛЕВУЮ и ПРАВУЮ камеры:", (10, 450),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(combined, "L - слева ЛЕВАЯ | R - слева ПРАВАЯ | Q - отмена", (10, 480),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)
        
        # ОТОБРАЖЕНИЕ КОМБИНИРОВАННОГО ИЗОБРАЖЕНИЯ
        cv2.imshow('Detect left/right camera', combined)
        
        # ОЖИДАНИЕ КЛАВИШИ (100 мс)
        key = cv2.waitKey(100) & 0xFF
        
        # ОБРАБОТКА КЛАВИШ:
        if key == ord('l'):
            # cam1 слева, cam2 справа — назначение корректно
            cv2.destroyAllWindows()
            return cam1_id, cam2_id, cap1, cap2
        elif key == ord('r'):
            # cam2 слева, cam1 справа — меняем местами
            cv2.destroyAllWindows()
            return cam2_id, cam1_id, cap2, cap1
        elif key == ord('q') or key == 27:
            # Отмена выбора
            break
        # Дополнительная проверка закрытия окна пользователем
        if cv2.getWindowProperty('Detect left/right camera', cv2.WND_PROP_VISIBLE) < 0:
            break
    
    # ОЧИСТКА РЕСУРСОВ ПРИ ВЫХОДЕ
    cv2.destroyAllWindows()
    cap1.release()
    cap2.release()
    return None, None, None, None

def select_cameras_interactive_visual():
    """
    Полный интерактивный процесс выбора стереопары:
    1. Поиск доступных камер
    2. Визуальный выбор двух камер
    3. Определение физического расположения (левая/правая)
    
    ВОЗВРАЩАЕТ:
    ----------
    tuple : (left_id, right_id, left_cap, right_cap) или (None, None, None, None)
        Полностью настроенная стереопара или None при ошибке
    """
    
    print("\n" + "=" * 70)
    print("ВИЗУАЛЬНЫЙ ВЫБОР СТЕРЕОКАМЕР")
    print("=" * 70)
    print("\nШАГ 1: Поиск доступных камер...")
    available_cameras = find_available_cameras()
    
    # Проверка минимального количества камер
    if len(available_cameras) < 2:
        print(f"\nНедостаточно камер! Найдено: {len(available_cameras)}")
        for cam in available_cameras:
            cam['cap'].release()
        return None, None, None, None
    
    print("\nШАГ 2: Визуальный выбор двух камер...")
    cam1_id, cam2_id, cap1, cap2 = select_cameras_visual(available_cameras)
    
    # Проверка успешности выбора
    if cam1_id is None or cam2_id is None or cap1 is None or cap2 is None:
        print("Выбор камер отменен.")
        return None, None, None, None
    
    print("\nШАГ 3: Определение левой и правой камеры...")
    # Передача уже открытых камер — КРИТИЧЕСКИ ВАЖНО для macOS (избегаем повторного открытия)
    left_id, right_id, left_cap, right_cap = determine_left_right_with_existing_caps(
        cam1_id, cam2_id, cap1, cap2
    )
    
    # Проверка успешности определения расположения
    if left_id is None or right_id is None or left_cap is None or right_cap is None:
        print("Определение расположения отменено.")
        return None, None, None, None
    
    # ФИНАЛЬНЫЙ ОТЧЕТ О ВЫБОРЕ
    print("\n" + "-" * 70)
    print("ВЫБОР ЗАВЕРШЕН")
    print(f"  Левая камера: ID {left_id}")
    print(f"  Правая камера: ID {right_id}")
    print("-" * 70)
    return left_id, right_id, left_cap, right_cap

# ============================================================================
# КАРТА ГЛУБИНЫ ЧЕРЕЗ СТЕРЕОЗРЕНИЕ
# ============================================================================

def create_depth_map_stereo_scaled(left_img, right_img, min_disp, num_disp, window_size):
    """
    Вычисление карты глубины методом стереозрения с использованием алгоритма SGBM (Semi-Global Block Matching).
    
    АЛГОРИТМ SGBM:
    1. Сравнивает блоки пикселей между левым и правым изображениями вдоль эпиполярных линий
    2. Ищет смещение (диспаритет) для каждого пикселя, минимизируя стоимость соответствия
    3. Применяет регуляризацию для подавления шума и заполнения разрывов
    
    ПАРАМЕТРЫ АЛГОРИТМА:
    - minDisp: минимальное смещение (обычно 0 для ректифицированных изображений)
    - numDisparities: максимальное смещение (должно быть кратно 16)
    - blockSize: размер блока сравнения (нечетное число 5-21)
    - P1, P2: штрафы за различия в диспаритетах соседних пикселей
    - uniquenessRatio: порог уникальности соответствия
    - speckleWindowSize/speckleRange: параметры удаления шума ("пятен")
    
    ПАРАМЕТРЫ:
    ----------
    left_img, right_img : np.ndarray
        Ректифицированные изображения левой и правой камер
    min_disp : int
        Минимальный диспаритет (обычно 0)
    num_disp : int
        Максимальный диспаритет (кратна 16)
    window_size : int
        Размер блока сравнения (нечетное число)
    
    ВОЗВРАЩАЕТ:
    ----------
    tuple : (disparity_normalized, disparity_raw, depth_colormap, confidence)
        disparity_normalized : np.ndarray (float32)
            Нормализованная карта диспаритета [0, 255]
        disparity_raw : np.ndarray (float32)
            Сырая карта диспаритета в пикселях
        depth_colormap : np.ndarray (uint8)
            Цветовая карта глубины для визуализации (COLORMAP_JET)
        confidence : np.ndarray (float32)
            Карта достоверности (1.0 = надежное значение, 0.0 = ненадежное)
    """
    
    # Приведение изображений к одинаковому размеру (защита)
    left_img, right_img = ensure_same_size(left_img, right_img)
    
    # ПРЕОБРАЗОВАНИЕ В ГРАДАЦИИ СЕРОГО ЕСЛИ ЦВЕТНЫЕ
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if len(left_img.shape) == 3 else left_img
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if len(right_img.shape) == 3 else right_img
    
    # ПРЕОБРАЗОВАНИЕ В UINT8 И ОГРАНИЧЕНИЕ ДИАПАЗОНА [0, 255]
    gray_left = np.uint8(np.clip(gray_left, 0, 255))
    gray_right = np.uint8(np.clip(gray_right, 0, 255))
    
    try:
        # ИНИЦИАЛИЗАЦИЯ АЛГОРИТМА SGBM С ОПТИМАЛЬНЫМИ ПАРАМЕТРАМИ:
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,                     # Минимальный диспаритет
            numDisparities=num_disp,                   # Максимальный диспаритет (кратна 16)
            blockSize=window_size,                     # Размер блока сравнения
            P1=8 * 3 * window_size ** 2,               # Штраф за небольшие различия в соседях
            P2=32 * 3 * window_size ** 2,              # Штраф за большие различия в соседях
            disp12MaxDiff=1,                           # Макс. различие между левым-правым поиском
            uniquenessRatio=10,                        # Порог уникальности соответствия (%)
            speckleWindowSize=100,                     # Размер окна для удаления шума
            speckleRange=32,                           # Макс. различие в диспаритетах в окне
            preFilterCap=63,                           # Макс. значение после предфильтрации
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY        # Использование трех направлений для регуляризации
        )
        
        # ВЫЧИСЛЕНИЕ КАРТЫ ДИСПАРИТЕТА:
        # Результат возвращается в формате int16 с масштабом 16 (для субпиксельной точности)
        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # МЕДИАННОЕ СГЛАЖИВАНИЕ ДЛЯ ПОДАВЛЕНИЯ ИМПУЛЬСНОГО ШУМА
        disparity = cv2.medianBlur(disparity, 5)
        
        # ОГРАНИЧЕНИЕ ДИАПАЗОНА ДИСПАРИТЕТА ДЛЯ СТАБИЛЬНОСТИ
        disparity_clipped = np.clip(disparity, min_disp, min_disp + num_disp - 1)
        
        # НОРМАЛИЗАЦИЯ В ДИАПАЗОН [0, 255] ДЛЯ ВИЗУАЛИЗАЦИИ
        disparity_normalized = ((disparity_clipped - min_disp) / num_disp * 255.0).astype(np.uint8)
        
        # СОЗДАНИЕ ЦВЕТОВОЙ КАРТЫ ДЛЯ ВИЗУАЛИЗАЦИИ (спектр от синего к красному)
        depth_colormap = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
        
        # ДОБАВЛЕНИЕ ИНФОРМАЦИИ О МАСШТАБЕ И ПАРАМЕТРАХ В ИЗОБРАЖЕНИЕ
        disp_info = f"Scale:{PROCESSING_SCALE:.2f}x Disp:{num_disp}px"
        cv2.putText(depth_colormap, disp_info, (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
        
        # ВЫЧИСЛЕНИЕ КАРТЫ ДОСТОВЕРНОСТИ:
        # Надежными считаются пиксели с диспаритетом в середине диапазона (не на границах)
        valid_mask = (disparity > min_disp + 1) & (disparity < min_disp + num_disp - 1)
        confidence = np.zeros_like(disparity, dtype=np.float32)
        confidence[valid_mask] = 1.0  # Полная достоверность для валидных пикселей
        
        return disparity_normalized.astype(np.float32), disparity, depth_colormap, confidence
        
    except Exception as e:
        # ОБРАБОТКА ОШИБОК С ГЕНЕРАЦИЕЙ ПУСТЫХ МАССИВОВ ДЛЯ ПРОДОЛЖЕНИЯ РАБОТЫ
        print(f"Ошибка стерео: {e}")
        import traceback
        traceback.print_exc()
        h, w = gray_left.shape[:2]
        empty_disp = np.zeros((h, w), dtype=np.float32)
        empty_disp_raw = np.zeros((h, w), dtype=np.float32)
        empty_colormap = cv2.applyColorMap(np.zeros((h, w), dtype=np.uint8), cv2.COLORMAP_JET)
        empty_conf = np.zeros((h, w), dtype=np.float32)
        return empty_disp, empty_disp_raw, empty_colormap, empty_conf

# ============================================================================
# MiDaS — НЕЙРОСЕТЕВАЯ ОЦЕНКА ГЛУБИНЫ (РЕЗЕРВНЫЙ МЕТОД)
# ============================================================================

class DepthEstimatorMidas:
    """
    Класс для оценки глубины с использованием предобученной нейросети MiDaS.
    
    ПРИНЦИП РАБОТЫ:
    - Использует сверточную нейросеть, обученную на множестве изображений с разметкой глубины
    - Выдает относительную карту глубины (не абсолютные метрические значения)
    - Работает с одной камерой — не требует стереопары
    
    ОСОБЕННОСТИ РЕАЛИЗАЦИИ:
    - Автоматическое определение устройства (CUDA/GPU или CPU)
    - Поддержка двух версий модели: "MiDaS_small" (быстрая) и "MiDaS" (точная)
    - Нормализация выходных данных в диапазон [0, 255]
    - Оценка достоверности через контраст изображения
    
    АТРИБУТЫ:
    ----------
    device : torch.device
        Устройство выполнения (cuda:0 или cpu)
    model : torch.nn.Module
        Загруженная модель MiDaS
    transform : callable
        Функция предобработки изображения для модели
    """
    
    def __init__(self, model_type="MiDaS_small"):
        """
        Инициализация оценщика глубины MiDaS.
        
        ПАРАМЕТРЫ:
        ----------
        model_type : str, optional (default="MiDaS_small")
            Тип модели: "MiDaS_small" (быстрая, менее точная) или "MiDaS" (точная, медленная)
        """
        # Проверка наличия PyTorch (выполняется на уровне модуля, но дублируем для надежности)
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch не установлен. MiDaS недоступен.")
        
        # ОПРЕДЕЛЕНИЕ УСТРОЙСТВА ВЫПОЛНЕНИЯ:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ИНФО] MiDaS: Используется устройство: {self.device}")
        
        # ЗАГРУЗКА МОДЕЛИ ИЗ РЕПОЗИТОРИЯ HUB:
        print(f"[ИНФО] MiDaS: Загрузка модели '{model_type}'...")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device)      # Перемещение модели на целевое устройство
        self.model.eval()               # Переключение в режим инференса (без градиентов)
        
        # ЗАГРУЗКА ТРАНСФОРМОВ ДЛЯ ПРЕДОБРАБОТКИ:
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        # Выбор трансформа в зависимости от типа модели
        self.transform = transforms.small_transform if model_type == "MiDaS_small" else transforms.default_transform
        
        print(f"[ИНФО] MiDaS: Модель загружена. Тип: {model_type}")
    
    def __call__(self, frame):
        """
        Вычисление карты глубины для одного кадра.
        
        ПАРАМЕТРЫ:
        ----------
        frame : np.ndarray
            Входное изображение в формате BGR (OpenCV)
        
        ВОЗВРАЩАЕТ:
        ----------
        tuple : (depth_normalized, confidence) или (None, None) при ошибке
            depth_normalized : np.ndarray (float32, [0, 255])
                Нормализованная карта глубины
            confidence : np.ndarray (float32, [0, 1])
                Карта достоверности на основе контраста изображения
        """
        # ПРОВЕРКА ВАЛИДНОСТИ ВХОДНОГО ИЗОБРАЖЕНИЯ
        if frame is None or frame.size == 0:
            return None, None
        
        try:
            # ПРЕОБРАЗОВАНИЕ В RGB (требование MiDaS)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ПРИМЕНЕНИЕ ТРАНСФОРМА (нормализация, изменение размера до 384x384 для small)
            input_batch = self.transform(img_rgb).to(self.device)
            
            # ИНФЕРЕНС БЕЗ ВЫЧИСЛЕНИЯ ГРАДИЕНТОВ (ускорение и экономия памяти)
            with torch.no_grad():
                prediction = self.model(input_batch)
                # ИНТЕРПОЛЯЦИЯ РЕЗУЛЬТАТА ДО РАЗМЕРА ИСХОДНОГО ИЗОБРАЖЕНИЯ:
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),    # Добавление канального измерения
                    size=frame.shape[:2],       # Целевой размер = размер кадра
                    mode="bicubic",             # Бикубическая интерполяция для гладкости
                    align_corners=False,
                ).squeeze()  # Удаление лишних измерений
            
            # ПРЕОБРАЗОВАНИЕ В NUMPY И УДАЛЕНИЕ С ПАМЯТИ GPU
            depth_raw = prediction.cpu().numpy()
            
            # НОРМАЛИЗАЦИЯ В ДИАПАЗОН [0, 255]:
            d_min = depth_raw.min()
            d_max = depth_raw.max()
            # Защита от деления на ноль при однородном изображении
            if (d_max - d_min) < 1e-6:
                depth_normalized = np.zeros_like(depth_raw)
            else:
                depth_normalized = (depth_raw - d_min) / (d_max - d_min + 1e-8) * 255.0
            
            # ВЫЧИСЛЕНИЕ ДОСТОВЕРНОСТИ ЧЕРЕЗ КОНТРАСТ ИЗОБРАЖЕНИЯ:
            # Высокий контраст → высокая достоверность (текстурированные области)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Размытие для выделения контраста
            contrast = cv2.absdiff(gray, blurred)        # Абсолютная разница = контраст
            confidence = cv2.normalize(contrast, None, 0, 1, cv2.NORM_MINMAX)  # Нормализация в [0, 1]
            
            return depth_normalized.astype(np.float32), confidence
            
        except Exception as e:
            # ОБРАБОТКА ОШИБОК С ЛОГИРОВАНИЕМ ТРЕЙСБЭКА
            print(f"[MiDaS ОШИБКА] {e}")
            import traceback
            traceback.print_exc()
            return None, None

def calibrate_midas_to_stereo(midas_depth, stereo_disparity, stereo_confidence):
    """
    Калибровка диапазона глубины MiDaS к диапазону стереодиспаритета для корректного фьюжна.
    
    ПРОБЛЕМА:
    MiDaS выдает относительные значения глубины (0 = близко, 255 = далеко), 
    тогда как стерео выдает диспаритет в пикселях (больше = ближе).
    Простое смешивание приведет к артефактам.
    
    РЕШЕНИЕ:
    1. Использовать области с высокой достоверностью стерео (>0.7) как опорные точки
    2. Вычислить линейное преобразование (масштаб + смещение) для выравнивания диапазонов
    3. Применить преобразование ко всей карте MiDaS
    
    МАТЕМАТИКА КАЛИБРОВКИ:
    Пусть:
      S = стереодиспаритет (опорные точки)
      M = значения MiDaS (опорные точки)
    
    Требуется найти параметры линейного преобразования:
      S_calibrated = scale * M + offset
    
    Решение методом наименьших квадратов через перцентили (устойчиво к выбросам):
      scale = (S_90% - S_10%) / (M_90% - M_10%)
      offset = S_10% - scale * M_10%
    
    ПАРАМЕТРЫ:
    ----------
    midas_depth : np.ndarray
        Сырая карта глубины от MiDaS (любой диапазон)
    stereo_disparity : np.ndarray
        Сырая карта диспаритета от стерео (пиксели)
    stereo_confidence : np.ndarray
        Карта достоверности стерео [0, 1]
    
    ВОЗВРАЩАЕТ:
    ----------
    np.ndarray или None
        Откалиброванная карта глубины MiDaS в диапазоне стереодиспаритета
    """
    
    # ПРОВЕРКА ВАЛИДНОСТИ ВХОДНЫХ ДАННЫХ
    if midas_depth is None or stereo_disparity is None:
        return None
    
    # ПРИВЕДЕНИЕ РАЗМЕРОВ К ОДИНАКОВЫМ (если различаются)
    if midas_depth.shape != stereo_disparity.shape:
        midas_depth = cv2.resize(midas_depth, (stereo_disparity.shape[1], stereo_disparity.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
    
    # ВЫДЕЛЕНИЕ ОПОРНЫХ ТОЧЕК С ВЫСОКОЙ ДОСТОВЕРНОСТЬЮ СТЕРЕО
    reliable_mask = stereo_confidence > 0.7
    
    # СЛУЧАЙ 1: НЕДОСТАТОЧНО ОПОРНЫХ ТОЧЕК — ИСПОЛЬЗУЕМ ПРОСТУЮ НОРМАЛИЗАЦИЮ ПО ПЕРЦЕНТИЛЯМ
    if np.sum(reliable_mask) < 100:
        # Вычисление перцентилей для устойчивости к выбросам
        midas_min = np.percentile(midas_depth, 5)
        midas_max = np.percentile(midas_depth, 95)
        stereo_min = np.percentile(stereo_disparity, 5)
        stereo_max = np.percentile(stereo_disparity, 95)
        
        # Защита от деления на ноль
        if (midas_max - midas_min) < 1e-6:
            return np.full_like(midas_depth, (stereo_min + stereo_max) / 2.0)
        
        # Линейная нормализация в диапазон стерео
        normalized = (midas_depth - midas_min) / (midas_max - midas_min + 1e-8)
        calibrated = stereo_min + normalized * (stereo_max - stereo_min)
        return calibrated.astype(np.float32)
    
    # СЛУЧАЙ 2: ДОСТАТОЧНО ОПОРНЫХ ТОЧЕК — ТОЧНАЯ КАЛИБРОВКА
    # Извлечение значений только в надежных областях
    stereo_vals = stereo_disparity[reliable_mask]
    midas_vals = midas_depth[reliable_mask]
    
    # Вычисление перцентилей 10% и 90% для устойчивости к выбросам
    stereo_min, stereo_max = np.percentile(stereo_vals, [10, 90])
    midas_min, midas_max = np.percentile(midas_vals, [10, 90])
    
    # Вычисление параметров линейного преобразования
    if (midas_max - midas_min) < 1e-6:
        scale = 1.0  # Защита от деления на ноль
    else:
        scale = (stereo_max - stereo_min) / (midas_max - midas_min + 1e-8)
    offset = stereo_min - midas_min * scale
    
    # Применение калибровки ко всей карте MiDaS
    calibrated = midas_depth * scale + offset
    return calibrated.astype(np.float32)

# ============================================================================
# ОПТИЧЕСКИЙ ПОТОК — СТРУКТУРНЫЙ АНАЛИЗ ДВИЖЕНИЯ (ДОПОЛНИТЕЛЬНЫЙ МЕТОД)
# ============================================================================

class OpticalFlowDepthEstimator:
    """
    Оценщик глубины на основе анализа оптического потока и движения камеры.
    
    ТЕОРЕТИЧЕСКАЯ ОСНОВА:
    При движении камеры:
    - Близкие объекты демонстрируют большой оптический поток
    - Дальние объекты (горизонт) демонстрируют малый поток
    - Эго-движение камеры создает предсказуемый паттерн потока (гомография)
    
    АЛГОРИТМ:
    1. Вычисление оптического потока между последовательными кадрами (Farneback)
    2. Оценка эго-движения камеры через гомографию (RANSAC)
    3. Вычисление остаточного потока (реальный поток - предсказанный эго-движением)
    4. Преобразование остаточного потока в оценку глубины: глубина ∝ 1/остаток
    
    ОСОБЕННОСТИ:
    - Работает только при движении камеры
    - Использует билиатеральную фильтрацию для сглаживания
    - Поддерживает экспоненциальное сглаживание для стабильности
    - Автоматическое определение состояния движения через таймаут
    
    АТРИБУТЫ:
    ----------
    min_inliers : int
        Минимальное число инлайеров для валидации гомографии
    motion_timeout : float
        Время в секундах, в течение которого камера считается движущейся после последнего движения
    prev_gray : np.ndarray
        Предыдущий кадр в градациях серого (для вычисления потока)
    last_move_time : float
        Время последнего обнаруженного движения камеры
    camera_moving : bool
        Флаг текущего состояния движения камеры
    stable_depth : np.ndarray
        Экспоненциально сглаженная карта глубины для статичных сцен
    """
    
    def __init__(self, min_inliers=15, motion_timeout=1.5):
        """
        Инициализация оценщика оптического потока.
        
        ПАРАМЕТРЫ:
        ----------
        min_inliers : int, optional (default=15)
            Минимальное число инлайеров для валидации гомографии движения
        motion_timeout : float, optional (default=1.5)
            Таймаут в секундах для определения статичной сцены после движения
        """
        self.min_inliers = min_inliers
        self.motion_timeout = motion_timeout
        self.prev_gray = None              # Предыдущий кадр (инициализируется при первом вызове)
        self.last_move_time = time.time()  # Время последнего движения
        self.camera_moving = False         # Флаг движения камеры
        self.stable_depth = None           # Сглаженная карта глубины
        
        print(f"[ИНФО] Инициализирован оценщик глубины на основе оптического потока")
        print(f"       Порог inliers: {min_inliers}, Таймаут движения: {motion_timeout} сек")
    
    def __call__(self, frame):
        """
        Вычисление карты глубины на основе оптического потока для текущего кадра.
        
        ПАРАМЕТРЫ:
        ----------
        frame : np.ndarray
            Текущий кадр в формате BGR
        
        ВОЗВРАЩАЕТ:
        ----------
        np.ndarray или None
            Карта глубины в относительных единицах или None при ошибке/статичной сцене
        """
        # ПРОВЕРКА ВАЛИДНОСТИ ВХОДНОГО ИЗОБРАЖЕНИЯ
        if frame is None or frame.size == 0:
            return None
        
        # ПРЕОБРАЗОВАНИЕ В ГРАДАЦИИ СЕРОГО
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ИНИЦИАЛИЗАЦИЯ ПРИ ПЕРВОМ ВЫЗОВЕ
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return None
        
        # ПРОВЕРКА СОВПАДЕНИЯ РАЗМЕРОВ С ПРЕДЫДУЩИМ КАДРОМ
        if gray.shape != self.prev_gray.shape:
            self.prev_gray = gray.copy()
            self.stable_depth = None
            return None
        
        # ОБНОВЛЕНИЕ СОСТОЯНИЯ ДВИЖЕНИЯ КАМЕРЫ:
        current_time = time.time()
        # Камера считается движущейся, если прошло меньше motion_timeout секунд с последнего движения
        self.camera_moving = (current_time - self.last_move_time) < self.motion_timeout
        
        try:
            # ВЫЧИСЛЕНИЕ ОПТИЧЕСКОГО ПОТОКА МЕТОДОМ FARNERBACK:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5,    # Масштаб пирамиды (0.5 = уменьшение в 2 раза на уровень)
                levels=3,         # Количество уровней пирамиды
                winsize=15,       # Размер окна для полиномиальной аппроксимации
                iterations=3,     # Количество итераций на каждом уровне
                poly_n=5,         # Размер окна для полиномиальной аппроксимации (5 или 7)
                poly_sigma=1.2,   # Стандартное отклонение Гауссова ядра для полинома
                flags=0           # Флаги (0 = стандартный режим)
            )
        except cv2.error:
            # Обработка ошибок OpenCV (например, при очень быстром движении)
            self.prev_gray = gray.copy()
            return None
        
        # ОЦЕНКА ЭГО-ДВИЖЕНИЯ КАМЕРЫ ЧЕРЕЗ ГОМОГРАФИЮ:
        ego_motion_valid, expected_flow = self._estimate_ego_motion(flow)
        
        if ego_motion_valid:
            # ОБНОВЛЕНИЕ ВРЕМЕНИ ПОСЛЕДНЕГО ДВИЖЕНИЯ
            self.last_move_time = current_time
            self.camera_moving = True
            
            # ВЫЧИСЛЕНИЕ ОСТАТОЧНОГО ПОТОКА:
            # residual = реальный поток - предсказанный эго-движением
            residual_flow_x = flow[..., 0] - expected_flow[..., 0]
            residual_flow_y = flow[..., 1] - expected_flow[..., 1]
            residual_magnitude = np.hypot(residual_flow_x, residual_flow_y)  # Модуль вектора
            
            # ПРЕОБРАЗОВАНИЕ ОСТАТКА В ГЛУБИНУ:
            # Глубина обратно пропорциональна остаточному потоку (ближе = больше остаток)
            current_depth = 1.0 / (residual_magnitude + 0.5)  # +0.5 для защиты от деления на ноль
            
            # ЭКСПОНЕНЦИАЛЬНОЕ СГЛАЖИВАНИЕ:
            if self.stable_depth is None:
                self.stable_depth = current_depth.copy()
            else:
                # При движении — быстрое обновление (0.9), при статике — медленное (0.99)
                alpha = 0.9 if self.camera_moving else 0.99
                self.stable_depth = alpha * self.stable_depth + (1 - alpha) * current_depth
            
            # БИЛИАТЕРАЛЬНАЯ ФИЛЬТРАЦИЯ ДЛЯ СГЛАЖИВАНИЯ БЕЗ РАЗМЫТИЯ ГРАНИЦ
            depth_smoothed = cv2.bilateralFilter(current_depth, 9, 75, 75)
            
            # ОБНОВЛЕНИЕ ПРЕДЫДУЩЕГО КАДРА И ВОЗВРАТ РЕЗУЛЬТАТА
            self.prev_gray = gray.copy()
            return depth_smoothed
        
        # СЛУЧАЙ: КАМЕРА ДВИЖЕТСЯ, НО ГОМОГРАФИЯ НЕ ВАЛИДНА (РЕЗКОЕ ДВИЖЕНИЕ)
        if self.camera_moving and self.stable_depth is not None:
            depth = cv2.bilateralFilter(self.stable_depth.copy(), 9, 75, 75)
            self.prev_gray = gray.copy()
            return depth
        
        # СЛУЧАЙ: СТАТИЧНАЯ СЦЕНА — НЕТ ИНФОРМАЦИИ О ГЛУБИНЕ ЧЕРЕЗ ПОТОК
        self.prev_gray = gray.copy()
        return None
    
    def _estimate_ego_motion(self, flow):
        """
        Оценка движения камеры (эго-движения) через вычисление гомографии между кадрами.
        
        АЛГОРИТМ:
        1. Выбор равномерно распределенных точек на изображении (сетка)
        2. Получение векторов потока для этих точек
        3. Вычисление гомографии между исходными и сдвинутыми точками (RANSAC)
        4. Валидация гомографии по числу инлайеров
        
        ПАРАМЕТРЫ:
        ----------
        flow : np.ndarray
            Поле оптического потока [H, W, 2]
        
        ВОЗВРАЩАЕТ:
        ----------
        tuple : (is_valid, expected_flow)
            is_valid : bool — валидна ли гомография
            expected_flow : np.ndarray — предсказанный поток от эго-движения
        """
        h, w = flow.shape[:2]  # Размеры изображения
        
        # ПРОВЕРКА МИНИМАЛЬНОГО РАЗМЕРА ДЛЯ ОЦЕНКИ
        if h < 50 or w < 50:
            return False, np.zeros_like(flow)
        
        # СОЗДАНИЕ СЕТКИ ТОЧЕК ДЛЯ ОЦЕНКИ (равномерное распределение)
        step = 16  # Шаг сетки в пикселях
        y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step]
        points = np.vstack((x_coords.ravel(), y_coords.ravel())).T.astype(np.float32)
        
        # ПРОВЕРКА НАЛИЧИЯ ТОЧЕК
        if len(points) == 0:
            return False, np.zeros_like(flow)
        
        # ПОЛУЧЕНИЕ ВЕКТОРОВ ПОТОКА ДЛЯ ТОЧЕК СЕТКИ
        try:
            flow_vectors = flow[y_coords.astype(int).ravel(), x_coords.astype(int).ravel()]
        except IndexError:
            return False, np.zeros_like(flow)
        
        # ВЫЧИСЛЕНИЕ ЦЕЛЕВЫХ КООРДИНАТ (исходные + вектор потока)
        next_points = points + flow_vectors
        
        # ПРОВЕРКА ДОСТАТОЧНОГО КОЛИЧЕСТВА ТОЧЕК ДЛЯ RANSAC
        if len(points) < self.min_inliers * 2:
            return False, np.zeros_like(flow)
        
        try:
            # ВЫЧИСЛЕНИЕ ГОМОГРАФИИ МЕТОДОМ RANSAC:
            H, mask = cv2.findHomography(
                points, next_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,  # Порог репроекции для инлайера (пиксели)
                maxIters=2000,              # Максимальное число итераций RANSAC
                confidence=0.995            # Уровень достоверности
            )
        except cv2.error:
            return False, np.zeros_like(flow)
        
        # ВАЛИДАЦИЯ РЕЗУЛЬТАТА ГОМОГРАФИИ
        if H is None or mask is None:
            return False, np.zeros_like(flow)
        
        inliers = np.sum(mask)  # Число инлайеров
        
        # ПРОВЕРКА МИНИМАЛЬНОГО ЧИСЛА ИНЛАЙЕРОВ
        if inliers < self.min_inliers:
            return False, np.zeros_like(flow)
        
        # ПОСТРОЕНИЕ ПОЛНОЙ КАРТЫ ПРЕДСКАЗАННОГО ПОТОКА:
        # 1. Создание сетки всех пикселей изображения
        coords = np.array([[x, y] for y in range(h) for x in range(w)], dtype=np.float32).reshape(-1, 1, 2)
        try:
            # 2. Применение гомографии ко всем пикселям
            warped_coords = cv2.perspectiveTransform(coords, H).reshape(-1, 2)
            # 3. Вычисление векторов потока как разницы между преобразованными и исходными координатами
            expected_flow_flat = warped_coords - coords.reshape(-1, 2)
            expected_flow = expected_flow_flat.reshape(h, w, 2)
            return True, expected_flow
        except Exception:
            return False, np.zeros_like(flow)

def normalize_to_stereo_range(depth_map, stereo_disparity):
    """
    Нормализация произвольной карты глубины к диапазону стереодиспаритета.
    
    Когда используется:
    - Оптического потока (выдает относительные значения)
    - MiDaS при отсутствии стерео для калибровки
    
    Алгоритм:
    1. Определение валидного диапазона стереодиспаритета (игнорирование нулевых значений)
    2. Вычисление перцентилей 5% и 95% для устойчивости к выбросам
    3. Линейное преобразование входной карты глубины в диапазон стерео
    
    ПАРАМЕТРЫ:
    ----------
    depth_map : np.ndarray
        Входная карта глубины (любой диапазон)
    stereo_disparity : np.ndarray
        Карта стереодиспаритета для определения целевого диапазона
    
    ВОЗВРАЩАЕТ:
    ----------
    np.ndarray или None
        Нормализованная карта глубины в диапазоне стереодиспаритета
    """
    # ПРОВЕРКА ВАЛИДНОСТИ ВХОДНЫХ ДАННЫХ
    if depth_map is None or stereo_disparity is None:
        return None
    
    # ВЫДЕЛЕНИЕ ВАЛИДНЫХ ПИКСЕЛЕЙ СТЕРЕО (диспаритет > 0)
    stereo_valid = stereo_disparity > 0
    
    # ОПРЕДЕЛЕНИЕ ДИАПАЗОНА СТЕРЕО ПО ВАЛИДНЫМ ПИКСЕЛЯМ
    if np.any(stereo_valid):
        stereo_min = np.percentile(stereo_disparity[stereo_valid], 5)
        stereo_max = np.percentile(stereo_disparity[stereo_valid], 95)
    else:
        # Резервный диапазон при отсутствии валидных пикселей
        stereo_min, stereo_max = 0, 255
    
    # ВЫЧИСЛЕНИЕ ПЕРЦЕНТИЛЕЙ ВХОДНОЙ КАРТЫ ГЛУБИНЫ
    d_min = np.percentile(depth_map, 5)
    d_max = np.percentile(depth_map, 95)
    
    # ЛИНЕЙНАЯ НОРМАЛИЗАЦИЯ С ЗАЩИТОЙ ОТ ДЕЛЕНИЯ НА НОЛЬ
    if (d_max - d_min) < 1e-6:
        normalized = np.full_like(depth_map, (stereo_min + stereo_max) / 2.0)
    else:
        normalized = (depth_map - d_min) / (d_max - d_min + 1e-8)
        normalized = stereo_min + normalized * (stereo_max - stereo_min)
    
    return normalized.astype(np.float32)

# ============================================================================
# ФУНКЦИИ ВИЗУАЛИЗАЦИИ И ФЬЮЖНА
# ============================================================================

def fuse_depth_maps(stereo_depth, stereo_conf,
                    midas_depth_calibrated, midas_conf,
                    flow_depth_normalized, camera_moving,
                    use_stereo=True, use_midas=True, use_flow=True):
    """
    Объединение (фьюжн) карт глубины от разных методов в единую карту.
    
    Стратегия:
    1. ОСНОВА: Стереодиспаритет (при наличии и валидности)
    2. ЗАПОЛНЕНИЕ НЕНАДЕЖНЫХ ОБЛАСТЕЙ: MiDaS (где стерео имеет низкую достоверность)
    3. ЗАПОЛНЕНИЕ "ДЫР": Оптический поток (где диспаритет близок к нулю или отсутствует)
    
    Перекрытие:
    - При перекрытии одной камеры — отключение стерео, использование только резервных методов
    - При движении камеры — активация оптического потока для заполнения
    - Динамические веса заполнения настраиваются пользователем
    
    ПАРАМЕТРЫ:
    ----------
    stereo_depth : np.ndarray
        Нормализованная карта стереодиспаритета [0, 255]
    stereo_conf : np.ndarray
        Карта достоверности стерео [0, 1]
    midas_depth_calibrated : np.ndarray
        Откалиброванная карта глубины MiDaS в диапазоне стерео
    midas_conf : np.ndarray
        Карта достоверности MiDaS [0, 1]
    flow_depth_normalized : np.ndarray
        Нормализованная карта глубины оптического потока
    camera_moving : bool
        Флаг движения камеры (активирует использование потока)
    use_stereo, use_midas, use_flow : bool
        Флаги активации соответствующих методов
    
    ВОЗВРАЩАЕТ:
    ----------
    tuple : (fused_uint8, fused_colormap) или None при ошибке
        fused_uint8 : np.ndarray (uint8)
            Итоговая карта глубины в диапазоне [0, 255]
        fused_colormap : np.ndarray (uint8)
            Цветовая карта для визуализации с наложенной информацией
    """
    
    # ПРОВЕРКА НАЛИЧИЯ ХОТЯ БЫ ОДНОГО ВАЛИДНОГО МЕТОДА
    valid_stereo = use_stereo and stereo_depth is not None and stereo_depth.size > 0
    valid_midas = use_midas and midas_depth_calibrated is not None and midas_depth_calibrated.size > 0
    valid_flow = use_flow and flow_depth_normalized is not None and flow_depth_normalized.size > 0 and camera_moving
    
    if not (valid_stereo or valid_midas or valid_flow):
        return None
    
    # ОПРЕДЕЛЕНИЕ РАЗМЕРА ИТОГОВОЙ КАРТЫ (по первому доступному методу)
    if valid_stereo:
        h, w = stereo_depth.shape[:2]
    elif valid_midas:
        h, w = midas_depth_calibrated.shape[:2]
    else:
        h, w = flow_depth_normalized.shape[:2]
    
    # ИНИЦИАЛИЗАЦИЯ ИТОГОВЫХ МАССИВОВ
    fused = np.zeros((h, w), dtype=np.float32)      # Итоговая карта глубины
    fused_conf = np.zeros((h, w), dtype=np.float32) # Итоговая достоверность
    mode_parts = []  # Список активных методов для отображения в интерфейсе
    
    # СЦЕНАРИЙ 1: СТЕРЕО ДОСТУПНО (ОСНОВНОЙ СЛУЧАЙ)
    if valid_stereo:
        fused = stereo_depth.copy()
        fused_conf = stereo_conf.copy() if stereo_conf is not None else np.ones((h, w), dtype=np.float32)
        fused *= fusion_params['stereo_weight']  # Применение веса стерео
        mode_parts.append(f"Stereo×{fusion_params['stereo_weight']:.1f}")
        
        # ЗАПОЛНЕНИЕ НЕНАДЕЖНЫХ ОБЛАСТЕЙ MiDaS:
        if valid_midas and FUSION_METHODS['use_midas_fill']:
            # Маска областей с низкой достоверностью стерео
            low_conf_mask = fused_conf < fusion_params['stereo_conf_threshold']
            if np.any(low_conf_mask):
                # Вычисление веса заполнения на основе недостоверности (с Гауссовым сглаживанием)
                fill_weight = (1.0 - fused_conf) * fusion_params['midas_fill_weight']
                fill_weight = cv2.GaussianBlur(
                    fill_weight, 
                    (FUSION_SMOOTHING['midas_blend_radius'], FUSION_SMOOTHING['midas_blend_radius']), 
                    0
                )
                fill_weight = np.clip(fill_weight, 0.0, 1.0)
                mask = low_conf_mask & (fill_weight > 0.1)
                if np.any(mask):
                    # Линейное смешивание: результат = стерео*(1-вес) + MiDaS*вес
                    fused[mask] = fused[mask] * (1 - fill_weight[mask]) + \
                                 midas_depth_calibrated[mask] * fill_weight[mask]
                    fused_conf[mask] = np.maximum(fused_conf[mask], 0.8)  # Повышение достоверности
                    mode_parts.append(f"MiDaS_fill×{fusion_params['midas_fill_weight']:.1f}")
        
        # ЗАПОЛНЕНИЕ "ДЫР" ОПТИЧЕСКИМ ПОТОКОМ:
        if valid_flow and FUSION_METHODS['use_flow_fill']:
            # Маска "дыр" — области с очень низким диспаритетом или нулевым значением
            hole_mask = (fused < fusion_params['flow_hole_threshold']) | (fused == 0)
            if np.any(hole_mask):
                flow_weight = fusion_params['flow_fill_weight']
                # Линейное смешивание с фиксированным весом
                fused[hole_mask] = fused[hole_mask] * (1 - flow_weight) + \
                                 flow_depth_normalized[hole_mask] * flow_weight
                fused_conf[hole_mask] = np.maximum(fused_conf[hole_mask], 0.6)
                mode_parts.append(f"Flow_fill×{fusion_params['flow_fill_weight']:.1f}")
    
    # СЦЕНАРИЙ 2: СТЕРЕО НЕДОСТУПНО, НО ДОСТУПЕН MiDaS
    elif valid_midas:
        fused = midas_depth_calibrated.copy()
        fused_conf = midas_conf.copy() if midas_conf is not None else np.ones((h, w), dtype=np.float32) * 0.8
        mode_parts.append("MiDaS_base")
        
        # ЗАПОЛНЕНИЕ "ДЫР" ПОТОКОМ (если доступен и камера движется)
        if valid_flow and FUSION_METHODS['use_flow_fill']:
            hole_mask = (fused < fusion_params['flow_hole_threshold']) | (fused == 0)
            if np.any(hole_mask):
                flow_weight = fusion_params['flow_fill_weight']
                fused[hole_mask] = fused[hole_mask] * (1 - flow_weight) + \
                                 flow_depth_normalized[hole_mask] * flow_weight
                fused_conf[hole_mask] = np.maximum(fused_conf[hole_mask], 0.7)
                mode_parts.append(f"Flow_fill×{fusion_params['flow_fill_weight']:.1f}")
    
    # СЦЕНАРИЙ 3: ТОЛЬКО ОПТИЧЕСКИЙ ПОТОК
    elif valid_flow:
        fused = flow_depth_normalized.copy()
        fused_conf = np.ones((h, w), dtype=np.float32) * 0.7
        mode_parts.append("Flow_base")
    
    # ФИНАЛЬНОЕ СГЛАЖИВАНИЕ БИЛИАТЕРАЛЬНЫМ ФИЛЬТРОМ (ПРИ ЗНАЧИТЕЛЬНЫХ ЗНАЧЕНИЯХ)
    if np.max(fused) > 10.0:
        fused = cv2.bilateralFilter(
            fused.astype(np.float32),
            FUSION_SMOOTHING['fusion_bilateral_d'],
            FUSION_SMOOTHING['fusion_bilateral_sigma'],
            FUSION_SMOOTHING['fusion_bilateral_sigma']
        )
    
    # ОГРАНИЧЕНИЕ ДИАПАЗОНА И ПРЕОБРАЗОВАНИЕ В UINT8
    fused = np.clip(fused, 0, 255)
    fused_uint8 = fused.astype(np.uint8)
    
    # СОЗДАНИЕ ЦВЕТОВОЙ КАРТЫ ДЛЯ ВИЗУАЛИЗАЦИИ
    fused_colormap = cv2.applyColorMap(fused_uint8, cv2.COLORMAP_JET)
    
    # НАЛОЖЕНИЕ ИНФОРМАЦИИ ОБ АКТИВНЫХ МЕТОДАХ
    mode_text = " + ".join(mode_parts) if mode_parts else "NO DATA"
    cv2.putText(fused_colormap, f"FUSED: {mode_text}", (10, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 255, 255), 2)
    
    # НАЛОЖЕНИЕ ДИАПАЗОНА ЗНАЧЕНИЙ ГЛУБИНЫ
    min_val, max_val = np.min(fused), np.max(fused)
    cv2.putText(fused_colormap, f"Range: {min_val:.0f}-{max_val:.0f}", (10, 55),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
    
    # ИНДИКАЦИЯ СОСТОЯНИЯ ДВИЖЕНИЯ КАМЕРЫ (ЕСЛИ ДОСТУПЕН ПОТОК)
    if valid_flow:
        motion_text = "CAM MOVING" if camera_moving else "STATIC"
        cv2.putText(fused_colormap, motion_text, (w - 150, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0) if camera_moving else (0, 0, 255), 1)
    
    return fused_uint8, fused_colormap

def put_multiline_text_anywhere(img, text, position='bottom', font_face=cv2.FONT_HERSHEY_COMPLEX,
                                font_scale=0.6, color=(255, 128, 0), thickness=2,
                                margin=20, line_spacing=30):
    """
    Уместить многострочный текст в изображение с автоматическим переносом слов.
    
    АЛГОРИТМ ПЕРЕНОСА:
    1. Разбивка текста на слова
    2. Формирование строк с максимальной шириной (с учетом отступов)
    3. Вертикальное позиционирование в зависимости от параметра position
    
    ПАРАМЕТРЫ:
    ----------
    img : np.ndarray
        Изображение для наложения текста (изменяется in-place)
    text : str
        Текст для отображения
    position : str, optional (default='bottom')
        Позиция текста: 'top', 'bottom', 'center'
    font_face : int, optional
        Тип шрифта OpenCV
    font_scale : float, optional (default=0.6)
        Масштаб шрифта
    color : tuple, optional (default=(255, 128, 0))
        Цвет текста в формате BGR
    thickness : int, optional (default=2)
        Толщина линий шрифта
    margin : int, optional (default=20)
        Отступ от краев изображения
    line_spacing : int, optional (default=30)
        Вертикальное расстояние между строками
    
    ВОЗВРАЩАЕТ:
    ----------
    int : Количество отображенных строк
    """
    img_height, img_width = img.shape[:2]
    max_width = img_width - 2 * margin  # Максимальная ширина строки с учетом отступов
    
    # РАЗБИВКА ТЕКСТА НА СЛОВА И ФОРМИРОВАНИЕ СТРОК
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        # Проверка, помещается ли слово в текущую строку
        test_line = ' '.join(current_line + [word])
        (text_width, _), _ = cv2.getTextSize(test_line, font_face, font_scale, thickness)
        
        if text_width <= max_width:
            current_line.append(word)
        else:
            # Сохранение текущей строки и начало новой
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    # Добавление последней строки
    if current_line:
        lines.append(' '.join(current_line))
    
    # ВЫЧИСЛЕНИЕ ВЕРТИКАЛЬНОЙ ПОЗИЦИИ НАЧАЛА ТЕКСТА
    total_height = len(lines) * line_spacing
    
    if position == 'top':
        start_y = margin + line_spacing
    elif position == 'bottom':
        start_y = img_height - margin - total_height
        if start_y < margin:
            start_y = margin
    elif position == 'center':
        start_y = (img_height - total_height) // 2
    else:
        start_y = margin
    
    # НАЛОЖЕНИЕ ТЕКСТА ПО СТРОКАМ
    for i, line in enumerate(lines):
        y = start_y + i * line_spacing
        if y > img_height - margin:
            break
        cv2.putText(img, line, (margin, y), font_face, font_scale, color, thickness)
    
    return len(lines)

def display_fusion_params_panel(occlusion_state='none', left_score=0.0, right_score=0.0):
    """
    Отображение панели параметров фьюжна и статуса перекрытия камер.
    
    СОДЕРЖИМОЕ ПАНЕЛИ:
    - Текущий статус перекрытия камер (с цветовой индикацией)
    - Оценки перекрытия для левой и правой камер
    - Текущие значения параметров фьюжна
    - Подсказки по управлению параметрами
    
    ПАРАМЕТРЫ:
    ----------
    occlusion_state : str, optional (default='none')
        Статус перекрытия: 'none', 'left', 'right', 'both'
    left_score, right_score : float, optional (default=0.0)
        Оценки перекрытия для левой и правой камер (0.0–1.0+)
    """
    # СОЗДАНИЕ ЧЕРНОГО ФОНА ДЛЯ ПАНЕЛИ (450x280 пикселей)
    panel = np.zeros((280, 450, 3), dtype=np.uint8)
    y_offset = 30  # Вертикальное смещение для первой строки текста
    
    # ЗАГОЛОВОК ПАНЕЛИ
    cv2.putText(panel, "FUSION PARAMETERS & OCCLUSION STATUS", (10, 25),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 2)
    
    # ОТОБРАЖЕНИЕ СТАТУСА ПЕРЕКРЫТИЯ С ЦВЕТОВОЙ ИНДИКАЦИЕЙ:
    if occlusion_state == 'none':
        occlusion_color = (0, 255, 0)  # Зеленый = норма
        occlusion_text = "✓ Обе камеры СВОБОДНЫ (стерео активно)"
    elif occlusion_state == 'left':
        occlusion_color = (0, 165, 255)  # Оранжевый = предупреждение
        occlusion_text = "⚠ ЛЕВАЯ камера ПЕРЕКРЫТА (стерео отключено)"
    elif occlusion_state == 'right':
        occlusion_color = (0, 165, 255)
        occlusion_text = "⚠ ПРАВАЯ камера ПЕРЕКРЫТА (стерео отключено)"
    else:
        occlusion_color = (0, 255, 255)  # Желтый = обе перекрыты
        occlusion_text = "ⓘ Обе камеры перекрыты (стерео работает)"
    
    cv2.putText(panel, occlusion_text, (10, y_offset),
                cv2.FONT_HERSHEY_COMPLEX, 0.55, occlusion_color, 1)
    y_offset += 30
    
    # ОТОБРАЖЕНИЕ ЧИСЛОВЫХ ОЦЕНОК ПЕРЕКРЫТИЯ
    cv2.putText(panel, f"Оценка перекрытия: L={left_score:.2f} | R={right_score:.2f}", (10, y_offset),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 30
    
    # ОТОБРАЖЕНИЕ ТЕКУЩИХ ПАРАМЕТРОВ ФЬЮЖНА
    params_info = [
        f"Stereo weight:   {fusion_params['stereo_weight']:.2f} (W/S)",
        f"MiDaS fill:      {fusion_params['midas_fill_weight']:.2f} (E/D)",
        f"Flow fill:       {fusion_params['flow_fill_weight']:.2f} (R/F)",
        f"Conf threshold:  {fusion_params['stereo_conf_threshold']:.2f} (T/G)",
        f"Hole threshold:  {fusion_params['flow_hole_threshold']:.1f} (Y/H)",
    ]
    
    for i, line in enumerate(params_info):
        # Разные цвета для основных и дополнительных параметров
        color = (100, 255, 100) if i < 3 else (255, 200, 100)
        cv2.putText(panel, line, (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
    
    # macOS FIX: установка пустого обработчика клика для предотвращения потери фокуса
    if sys.platform == 'darwin':
        cv2.namedWindow('Fusion Parameters', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Fusion Parameters', lambda *args: None)
    
    # ОТОБРАЖЕНИЕ ПАНЕЛИ
    cv2.imshow('Fusion Parameters', panel)

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ — ТОЧКА ВХОДА В ПРОГРАММУ
# ============================================================================

def main():
    """
    ГЛАВНАЯ ФУНКЦИЯ СИСТЕМЫ ФЬЮЖНА КАРТ ГЛУБИНЫ
    
    АРХИТЕКТУРА РАБОТЫ:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 1. ИНТЕРАКТИВНАЯ ИНИЦИАЛИЗАЦИЯ СТЕРЕОПАРЫ                           │
    │    ├─ Поиск доступных камер в системе                               │
    │    ├─ Визуальный выбор двух камер через графический интерфейс       │
    │    └─ Определение физического расположения (левая/правая)           │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 2. ЗАГРУЗКА КАЛИБРОВКИ И ПОДГОТОВКА РЕКТИФИКАЦИИ                    │
    │    ├─ Загрузка параметров калибровки из файла                       │
    │    ├─ Масштабирование матриц камер под текущий масштаб обработки    │
    │    └─ Генерация карт перемещения пикселей для ректификации          │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 3. ИНИЦИАЛИЗАЦИЯ МЕТОДОВ ОЦЕНКИ ГЛУБИНЫ                             │
    │    ├─ Стереозрение (SGBM) — основной метод                          │
    │    ├─ MiDaS (нейросеть) — резервный метод                           │
    │    └─ Оптический поток — заполнение "дыр" при движении              │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 4. ГЛАВНЫЙ ЦИКЛ ОБРАБОТКИ КАДРОВ                                    │
    │    ├─ Чтение кадров с камер                                         │
    │    ├─ Применение ректификации                                       │
    │    ├─ Обнаружение перекрытия камер (гистерезисная логика)           │
    │    ├─ Параллельные вычисления глубины (пул потоков)                 │
    │    ├─ Калибровка диапазонов между методами                          │
    │    ├─ Адаптивный фьюжн с динамическими весами                       │
    │    ├─ Визуализация результатов                                      │
    │    └─ Обработка пользовательского ввода в реальном времени          │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    # ============================================================================
    # ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И НАСТРОЙКА ОКРУЖЕНИЯ
    # ============================================================================
    
    # Разрешение изменения глобальной переменной масштаба обработки изнутри функции
    global PROCESSING_SCALE
    
    # ВЫВОД ЗАГОЛОВКА ПРОГРАММЫ С ВЕРСИЕЙ И ПЛАТФОРМОЙ
    print("=" * 70)
    print("СИСТЕМА АДАПТИВНОГО ФЬЮЖНА С ОБНАРУЖЕНИЕМ ПЕРЕКРЫТИЯ КАМЕР v2.2 (macOS-оптимизированная)")
    print("=" * 70)
    print(f"\nМАСШТАБИРОВАНИЕ ОБРАБОТКИ: {PROCESSING_SCALE:.2f}x")
    print(f"Ожидаемое ускорение относительно полного разрешения: ~{1/PROCESSING_SCALE**2:.1f}×")
    
    # ============================================================================
    # ЭТАП 1: ИНТЕРАКТИВНЫЙ ВЫБОР СТЕРЕОПАРЫ
    # ============================================================================
    
    # ВЫЗОВ ФУНКЦИИ ВИЗУАЛЬНОГО ВЫБОРА КАМЕР:
    # ВАЖНО: возвращаются УЖЕ ОТКРЫТЫЕ объекты камер (критично для macOS — избегаем повторного открытия)
    # Функция возвращает кортеж: (левый_ID, правый_ID, объект_левой_камеры, объект_правой_камеры)
    LEFT_CAMERA_ID, RIGHT_CAMERA_ID, left_cap, right_cap = select_cameras_interactive_visual()
    
    # ПРОВЕРКА УСПЕШНОСТИ ВЫБОРА КАМЕР:
    # Если хотя бы один элемент равен None — выбор не удался, завершаем программу
    if LEFT_CAMERA_ID is None or RIGHT_CAMERA_ID is None or left_cap is None or right_cap is None:
        print("❌ Не удалось выбрать камеры. Выход из программы.")
        return  # Досрочный выход из функции main()
    
    # ПОДТВЕРЖДЕНИЕ УСПЕШНОГО ВЫБОРА С ИНФОРМАЦИЕЙ ОБ ИСПОЛЬЗУЕМЫХ КАМЕРАХ
    print(f"\n✅ Используем уже открытые камеры: Левая={LEFT_CAMERA_ID}, Правая={RIGHT_CAMERA_ID}")
    print("   ВАЖНО: камеры остаются открытыми без повторного открытия (стабильность на macOS)")
    
    # ============================================================================
    # ЭТАП 2: ЗАГРУЗКА КАЛИБРОВКИ И ПОДГОТОВКА РЕКТИФИКАЦИИ
    # ============================================================================
    
    # ЗАГРУЗКА ДАННЫХ КАЛИБРОВКИ С УЧЕТОМ ТЕКУЩЕГО МАСШТАБА ОБРАБОТКИ:
    # Функция возвращает словарь с параметрами ректификации или None при ошибке
    stereo_calib = load_stereo_calibration_with_scaling(scale_factor=PROCESSING_SCALE)
    
    # КРИТИЧЕСКАЯ ПРОВЕРКА УСПЕШНОСТИ ЗАГРУЗКИ КАЛИБРОВКИ:
    if stereo_calib is None:
        print("❌ Критическая ошибка: не удалось загрузить калибровку с ректификацией!")
        # Освобождение ресурсов камер перед выходом
        left_cap.release()
        right_cap.release()
        return  # Досрочный выход из функции main()
    
    # ВЫВОД ИНФОРМАЦИИ О РАЗРЕШЕНИЯХ ДЛЯ ПОЛЬЗОВАТЕЛЯ:
    target_size_orig = stereo_calib['img_size_orig']  # Оригинальное разрешение калибровки
    print(f"Оригинальное разрешение калибровки: {target_size_orig[0]}x{target_size_orig[1]} пикселей")
    proc_size = stereo_calib['img_size_proc']  # Разрешение для обработки после масштабирования
    print(f"Разрешение обработки (с ректификацией): {proc_size[0]}x{proc_size[1]} пикселей")
    
    # ПРОВЕРКА ФАКТИЧЕСКИХ РАЗРЕШЕНИЙ КАМЕР В РЕАЛЬНОМ ВРЕМЕНИ:
    # Получение текущих параметров через CAP_PROP_* (без изменения настроек!)
    left_width = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    left_height = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    right_width = int(right_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    right_height = int(right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ВЫВОД ФАКТИЧЕСКИХ РАЗРЕШЕНИЙ ДЛЯ ДИАГНОСТИКИ
    print(f"Фактическое разрешение левой камеры: {left_width}x{left_height}")
    print(f"Фактическое разрешение правой камеры: {right_width}x{right_height}")
    print("   Примечание: при несовпадении разрешений будет выполнено программное масштабирование")
    
    # ============================================================================
    # АДАПТАЦИЯ ПАРАМЕТРОВ СТЕРЕОАЛГОРИТМА ПОД МАСШТАБ ОБРАБОТКИ
    # ============================================================================
    
    # МАСШТАБИРОВАНИЕ ПАРАМЕТРА numDisparities (должен быть кратен 16 для OpenCV):
    # Умножаем базовое значение на коэффициент масштаба
    # Округляем вниз до ближайшего числа, кратного 16
    # Ограничиваем минимальным значением 16 (требование OpenCV)
    num_disp_scaled = max(16, int(NUM_DISP_BASE * PROCESSING_SCALE) // 16 * 16)
    
    # МАСШТАБИРОВАНИЕ ПАРАМЕТРА blockSize (должен быть нечетным числом):
    # Умножаем базовое значение на коэффициент масштаба
    # Округляем до ближайшего целого
    # Если результат четный — увеличиваем на 1 для получения нечетного числа
    window_size_scaled = max(5, int(WINDOW_SIZE_BASE * PROCESSING_SCALE))
    if window_size_scaled % 2 == 0:  # Проверка на четность
        window_size_scaled += 1      # Преобразование в нечетное число
    
    # ВЫВОД АДАПТИРОВАННЫХ ПАРАМЕТРОВ С ПОДТВЕРЖДЕНИЕМ СООТВЕТСТВИЯ ТРЕБОВАНИЯМ OpenCV
    print(f"\n[СТЕРЕО] Параметры адаптированы под масштаб {PROCESSING_SCALE:.2f}x:")
    print(f"  numDisparities: {NUM_DISP_BASE} → {num_disp_scaled} (кратно 16 ✅)")
    print(f"  blockSize: {WINDOW_SIZE_BASE} → {window_size_scaled} (нечётное ✅)")
    print(f"  minDisparity: {MIN_DISP_BASE} (без изменений)")
    
    # ============================================================================
    # ЭТАП 3: ИНИЦИАЛИЗАЦИЯ РЕЗЕРВНЫХ МЕТОДОВ ОЦЕНКИ ГЛУБИНЫ
    # ============================================================================
    
    # ИНИЦИАЛИЗАЦИЯ MiDaS (ЕСЛИ ДОСТУПЕН PyTorch):
    midas_estimator = None  # Инициализация как None на случай ошибки загрузки
    if TORCH_AVAILABLE:  # Проверка флага доступности PyTorch (установленного на этапе импорта)
        try:
            # Создание экземпляра оценщика глубины с моделью MiDaS_small (быстрая версия)
            midas_estimator = DepthEstimatorMidas(model_type="MiDaS_small")
            print("[ИНФО] MiDaS успешно инициализирован и готов к работе")
        except Exception as e:
            # Обработка ошибок инициализации (например, проблемы с загрузкой модели)
            print(f"[ВНИМАНИЕ] Не удалось инициализировать MiDaS: {e}")
            midas_estimator = None  # Сброс в None при ошибке
    
    # ИНИЦИАЛИЗАЦИЯ ОПТИЧЕСКОГО ПОТОКА:
    # Создание экземпляра оценщика с параметрами из глобальных конфигураций
    flow_estimator = OpticalFlowDepthEstimator(
        min_inliers=FUSION_THRESHOLDS['flow_min_inliers'],  # Минимальное число инлайеров для валидации гомографии
        motion_timeout=1.5  # Таймаут в секундах для определения статичной сцены
    )
    
    # ИНИЦИАЛИЗАЦИЯ ПУЛА ПОТОКОВ ДЛЯ ПАРАЛЛЕЛЬНОЙ ОБРАБОТКИ:
    # Создание пула с 2 рабочими потоками (оптимально для 2 основных методов: стерео + MiDaS)
    executor = ThreadPoolExecutor(max_workers=2)
    print("[ИНФО] ThreadPoolExecutor запущен (2 рабочих потока для параллельной обработки стерео и MiDaS)")
    
    # ============================================================================
    # ИНИЦИАЛИЗАЦИЯ ФЛАГОВ УПРАВЛЕНИЯ И СОСТОЯНИЯ СИСТЕМЫ
    # ============================================================================
    
    # ФЛАГИ АКТИВАЦИИ МЕТОДОВ И ВИЗУАЛИЗАЦИИ:
    use_stereo_rectify = True      # Применять ли ректификацию к изображениям (геометрическая коррекция)
    USE_STEREO = True              # Активировать ли вычисления стереоглубины
    show_depth = False             # Показывать ли карту глубины стерео
    USE_MIDAS = TORCH_AVAILABLE and (midas_estimator is not None)  # Активировать ли MiDaS (только если доступен)
    USE_OPTICAL_FLOW = True        # Активировать ли оптический поток
    show_midas_depth = False       # Показывать ли карту глубины MiDaS
    show_flow_depth = False        # Показывать ли карту глубины оптического потока
    show_fused_depth = True        # Показывать ли итоговую фьюжн-карту глубины
    show_fusion_panel = True       # Показывать ли панель параметров фьюжна
    
    # ============================================================================
    # КРИТИЧЕСКИ ВАЖНАЯ ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ ДЕТЕКТОРА ПЕРЕКРЫТИЯ
    # ============================================================================
    
    # ТЕКУЩЕЕ СОСТОЯНИЕ ПЕРЕКРЫТИЯ КАМЕР:
    # Возможные значения: 'none' (нет перекрытия), 'left' (левая перекрыта), 
    # 'right' (правая перекрыта), 'both' (обе перекрыты)
    occlusion_state = 'none'
    
    # СЧЕТЧИК ГИСТЕРЕЗИСА ДЛЯ ПОДАВЛЕНИЯ ЛОЖНЫХ СРАБАТЫВАНИЙ:
    # Требуется несколько последовательных кадров с одинаковым состоянием для подтверждения изменения
    occlusion_hysteresis_counter = 0
    
    # ПОРОГ ПОДТВЕРЖДЕНИЯ ИЗМЕНЕНИЯ СОСТОЯНИЯ (в кадрах):
    # Сколько последовательных кадров с новым состоянием требуется для его принятия
    OCCLUSION_HYSTERESIS_FRAMES = 5
    
    # ИНТЕРВАЛ ПРОВЕРКИ ПЕРЕКРЫТИЯ (в кадрах):
    # Проверка выполняется не на каждом кадре для оптимизации производительности
    OCCLUSION_CHECK_INTERVAL = 2
    
    # ============================================================================
    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ИНИЦИАЛИЗАЦИЯ ОЦЕНОК ПЕРЕКРЫТИЯ ДО ЦИКЛА!
    # ============================================================================
    
    # ИНИЦИАЛИЗАЦИЯ ОЦЕНОК ПЕРЕКРЫТИЯ ДЛЯ ЛЕВОЙ И ПРАВОЙ КАМЕР:
    # Эти переменные используются в основном цикле и должны быть определены ДО его начала
    # для избежания ошибки UnboundLocalError при первом кадре
    left_score = 0.0   # Оценка перекрытия левой камеры (0.0 = полностью свободна)
    right_score = 0.0  # Оценка перекрытия правой камеры (0.0 = полностью свободна)
    # Примечание: значения в диапазоне [0.0, 1.0+] где >0.45 обычно означает перекрытие
    
    # ============================================================================
    # ПАРАМЕТРЫ ПРОИЗВОДИТЕЛЬНОСТИ И СТАТИСТИКИ
    # ============================================================================
    
    # ПРОПУСК КАДРОВ ДЛЯ УСКОРЕНИЯ ОБРАБОТКИ:
    # Обработка только каждого N-го кадра (остальные пропускаются без вычислений)
    SKIP_FRAMES = 3
    
    # СЧЕТЧИК ВСЕХ ПРОЧИТАННЫХ КАДРОВ (включая пропущенные)
    frame_counter = 0
    
    # ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ ДЛЯ СБОРА СТАТИСТИКИ ПРОИЗВОДИТЕЛЬНОСТИ:
    last_stats_time = time.time()    # Время последнего вывода статистики
    total_frames = 0                 # Счетчик обработанных кадров (не пропущенных)
    stereo_time_sum = 0.0            # Суммарное время обработки стерео за период
    midas_time_sum = 0.0             # Суммарное время обработки MiDaS за период
    flow_time_sum = 0.0              # Суммарное время обработки оптического потока за период
    
    # ============================================================================
    # ВЫВОД ИНСТРУКЦИИ ПО УПРАВЛЕНИЮ ДЛЯ ПОЛЬЗОВАТЕЛЯ
    # ============================================================================
    
    print("\n" + "=" * 70)
    print("ИНСТРУКЦИЯ ПО УПРАВЛЕНИЮ В РЕАЛЬНОМ ВРЕМЕНИ")
    print("=" * 70)
    print("'q'  - выход из программы")
    print("'c'  - сохранить текущие кадры и карты глубины в папку depth_captures/")
    print("'v'  - переключить применение стереоректификации (ВКЛ/ВЫКЛ)")
    print("'1'  - переключить визуализацию карты глубины Стерео")
    print("'2'  - переключить визуализацию карты глубины MiDaS")
    print("'3'  - переключить визуализацию карты глубины оптического потока")
    print("'4'  - переключить ИТОГОВУЮ фьюжн-карту глубины (рекомендуется)")
    print("'5'  - показать/скрыть панель параметров фьюжна")
    print("'z'  - ВКЛ/ВЫКЛ метод СТЕРЕОЗРЕНИЯ (останавливает вычисления)")
    print("'m'  - ВКЛ/ВЫКЛ метод MiDaS (только если доступен PyTorch)")
    print("'o'  - ВКЛ/ВЫКЛ метод оптического потока")
    print("'p'  - сбросить все параметры фьюжна к значениям по умолчанию")
    print("'w/s' - увеличить/уменьшить вес стерео (диапазон: 0.1–1.5)")
    print("'e/d' - увеличить/уменьшить вес заполнения MiDaS (диапазон: 0.0–1.0)")
    print("'r/f' - увеличить/уменьшить вес заполнения потоком (диапазон: 0.0–1.0)")
    print("'t/g' - увеличить/уменьшить порог достоверности стерео (диапазон: 0.1–0.95)")
    print("'y/h' - увеличить/уменьшить порог 'дыр' для потока (диапазон: 1.0–50.0)")
    print("'+/-' - увеличить/уменьшить масштаб обработки (диапазон: 0.3–1.0)")
    print("\n✅ СИСТЕМА АВТОМАТИЧЕСКИ:")
    print("   • Обнаруживает перекрытие одной камеры и отключает стерео")
    print("   • Переключается на резервные методы (MiDaS + оптический поток)")
    print("   • Восстанавливает стерео при освобождении камеры")
    print("   • Применяет гистерезис для подавления ложных срабатываний")
    
    # macOS
    if sys.platform == 'darwin':
        print("\nⓘ macOS: Если управление клавиатурой перестало работать после")
        print("         перетаскивания окна — кликните ЛКМ внутри любого окна OpenCV")
    
    # ============================================================================
    # ГЛАВНЫЙ ЦИКЛ ОБРАБОТКИ КАДРОВ
    # ============================================================================
    
    try:  # Блок для перехвата исключений с корректной очисткой ресурсов
        while True:  # Бесконечный цикл обработки кадров
            
            # ЗАМЕР ВРЕМЕНИ НАЧАЛА ОБРАБОТКИ ТЕКУЩЕГО КАДРА
            frame_start = time.time()
            
            # ИНКРЕМЕНТ СЧЕТЧИКА ВСЕХ КАДРОВ (включая пропущенные)
            frame_counter += 1
            
            # ============================================================================
            # ПРОПУСК КАДРОВ ДЛЯ ОПТИМИЗАЦИИ ПРОИЗВОДИТЕЛЬНОСТИ
            # ============================================================================
            
            # ПРОВЕРКА УСЛОВИЯ ПРОПУСКА КАДРА:
            # Обрабатываем только каждый SKIP_FRAMES-й кадр (например, каждый 3-й)
            if frame_counter % SKIP_FRAMES != 0:
                # Чтение кадров без обработки (для синхронизации буферов камер)
                ret_left, frame_left = left_cap.read()
                ret_right, frame_right = right_cap.read()
                
                # ПРОВЕРКА УСПЕШНОСТИ ЧТЕНИЯ КАДРОВ:
                if not ret_left or not ret_right:
                    print("❌ Ошибка: Не удалось получить кадр с одной из камер")
                    break  # Выход из цикла при критической ошибке
                
                # ОЖИДАНИЕ КЛАВИШИ С УВЕЛИЧЕННОЙ ЗАДЕРЖКОЙ ДЛЯ macOS (10 мс вместо 1 мс)
                key = cv2.waitKey(10) & 0xFF
                
                # ПРОВЕРКА ЗАПРОСА НА ВЫХОД (клавиша 'q')
                if key == ord('q'):
                    break  # Выход из цикла обработки
                
                # Пропуск остальной обработки для данного кадра
                continue
            
            # ============================================================================
            # ИНКРЕМЕНТ СЧЕТЧИКА ОБРАБОТАННЫХ КАДРОВ
            # ============================================================================
            
            # Увеличение счетчика только для кадров, прошедших полную обработку
            total_frames += 1
            
            # ============================================================================
            # ЧТЕНИЕ КАДРОВ С ОБЕИХ КАМЕР
            # ============================================================================
            
            # Чтение кадра с левой камеры: возвращает (успех, изображение)
            ret_left, frame_left = left_cap.read()
            # Чтение кадра с правой камеры: возвращает (успех, изображение)
            ret_right, frame_right = right_cap.read()
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА УСПЕШНОСТИ ЧТЕНИЯ КАДРОВ:
            if not ret_left or not ret_right:
                print("❌ Ошибка: Не удалось получить кадр с одной из камер")
                break  # Выход из цикла при критической ошибке
            
            # ============================================================================
            # ЗАЩИТА ОТ БИТЫХ/ПУСТЫХ КАДРОВ (КРИТИЧЕСКИ ВАЖНО ДЛЯ macOS)
            # ============================================================================
            
            # Многоуровневая проверка валидности кадров:
            if (frame_left is None or frame_right is None or           # Проверка на None
                frame_left.size == 0 or frame_right.size == 0 or       # Проверка на пустой массив
                frame_left.shape[0] == 0 or frame_left.shape[1] == 0 or  # Проверка на нулевые размеры
                frame_right.shape[0] == 0 or frame_right.shape[1] == 0):
                print("⚠️  Получен некорректный кадр (битый/пустой). Пропускаем кадр.")
                time.sleep(0.1)  # Короткая пауза для стабилизации
                continue  # Пропуск текущего кадра
            
            # ============================================================================
            # ПРИМЕНЕНИЕ СТЕРЕОРЕКТИФИКАЦИИ (ЕСЛИ ВКЛЮЧЕНА)
            # ============================================================================
            
            # Проверка флага применения ректификации и наличия данных калибровки
            if use_stereo_rectify and stereo_calib is not None:
                # Применение ректификации к обеим камерам:
                # 1. Исправление дисторсии линз
                # 2. Выравнивание эпиполярных линий в горизонтальные прямые
                processed_left, processed_right = apply_stereo_rectification(
                    frame_left,        # Исходное изображение левой камеры
                    frame_right,       # Исходное изображение правой камеры
                    stereo_calib       # Словарь с параметрами ректификации
                )
            else:
                # РЕЖИМ БЕЗ РЕКТИФИКАЦИИ (только масштабирование):
                # Получение целевого разрешения из калибровки или значение по умолчанию
                proc_w, proc_h = stereo_calib['img_size_proc'] if stereo_calib else (640, 480)
                
                # Масштабирование изображений до целевого разрешения
                processed_left = cv2.resize(
                    frame_left, 
                    (proc_w, proc_h), 
                    interpolation=cv2.INTER_LINEAR  # Линейная интерполяция для качества
                )
                processed_right = cv2.resize(
                    frame_right, 
                    (proc_w, proc_h), 
                    interpolation=cv2.INTER_LINEAR
                )
            
            # ============================================================================
            # ЭТАП ОБНАРУЖЕНИЯ ПЕРЕКРЫТИЯ КАМЕР (С ГИСТЕРЕЗИСОМ)
            # ============================================================================
            
            # ПРОВЕРКА УСЛОВИЯ ВЫПОЛНЕНИЯ ПРОВЕРКИ ПЕРЕКРЫТИЯ:
            # Выполняется не на каждом кадре для оптимизации производительности
            if frame_counter % OCCLUSION_CHECK_INTERVAL == 0:
                # ВЫЗОВ ФУНКЦИИ ОБНАРУЖЕНИЯ ПЕРЕКРЫТИЯ:
                # Возвращает кортеж: (новое_состояние, оценка_левой, оценка_правой)
                new_occlusion_state, left_score, right_score = detect_camera_occlusion(
                    processed_left,      # Ректифицированное изображение левой камеры
                    processed_right,     # Ректифицированное изображение правой камеры
                    occlusion_threshold=0.45  # Порог принятия решения о перекрытии
                )
                
                # ============================================================================
                # ГИСТЕРЕЗИСНАЯ ЛОГИКА ДЛЯ ПОДАВЛЕНИЯ ЛОЖНЫХ СРАБАТЫВАНИЙ
                # ============================================================================
                
                # СРАВНЕНИЕ НОВОГО СОСТОЯНИЯ С ТЕКУЩИМ:
                if new_occlusion_state == occlusion_state:
                    # Состояние не изменилось — сброс счетчика гистерезиса
                    occlusion_hysteresis_counter = 0
                else:
                    # Состояние изменилось — инкремент счетчика подтверждения
                    occlusion_hysteresis_counter += 1
                    
                    # ПРОВЕРКА ДОСТИЖЕНИЯ ПОРОГА ПОДТВЕРЖДЕНИЯ ИЗМЕНЕНИЯ:
                    if occlusion_hysteresis_counter >= OCCLUSION_HYSTERESIS_FRAMES:
                        # ПОДТВЕРЖДЕНИЕ ИЗМЕНЕНИЯ СОСТОЯНИЯ:
                        occlusion_state = new_occlusion_state  # Обновление текущего состояния
                        occlusion_hysteresis_counter = 0       # Сброс счетчика
                        
                        # ============================================================================
                        # РЕАКЦИЯ НА ПЕРЕКРЫТИЕ ОДНОЙ КАМЕРЫ
                        # ============================================================================
                        
                        # СЛУЧАЙ: ПЕРЕКРЫТИЕ ТОЛЬКО ЛЕВОЙ ИЛИ ТОЛЬКО ПРАВОЙ КАМЕРЫ:
                        if occlusion_state in ['left', 'right']:
                            # Проверка, что стерео сейчас активно (чтобы избежать повторных сообщений)
                            if USE_STEREO:
                                # ВЫВОД ПРЕДУПРЕЖДЕНИЯ В КОНСОЛЬ:
                                print(f"\n⚠️  ОБНАРУЖЕНО ПЕРЕКРЫТИЕ {occlusion_state.upper()} КАМЕРЫ!")
                                print(f"    Стереозрение автоматически отключено.")
                                print(f"    Активирован резервный режим: только {'правая' if occlusion_state == 'left' else 'левая'} камера + MiDaS + оптический поток")
                                # АВТОМАТИЧЕСКОЕ ОТКЛЮЧЕНИЕ СТЕРЕО:
                                USE_STEREO = False
                        
                        # ============================================================================
                        # РЕАКЦИЯ НА ВОССТАНОВЛЕНИЕ ОБЗОРА ИЛИ ПЕРЕКРЫТИЕ ОБЕИХ КАМЕР
                        # ============================================================================
                        
                        # СЛУЧАЙ: ОБЕ КАМЕРЫ СВОБОДНЫ ИЛИ ОБЕ ПЕРЕКРЫТЫ:
                        # Примечание: при перекрытии обеих камер стерео всё равно может работать (хотя качество снижено)
                        elif occlusion_state in ['none', 'both']:
                            # Проверка, что стерео сейчас отключено
                            if not USE_STEREO:
                                # ВЫВОД ПОДТВЕРЖДЕНИЯ ВОССТАНОВЛЕНИЯ:
                                status_text = "ВОССТАНОВЛЕНИЕ ОБЗОРА" if occlusion_state == 'none' else "ОБЕ КАМЕРЫ ПЕРЕКРЫТЫ (стерео всё равно работает)"
                                print(f"\n✅ {status_text}!")
                                print("    Стереозрение автоматически включено.")
                                # АВТОМАТИЧЕСКОЕ ВКЛЮЧЕНИЕ СТЕРЕО:
                                USE_STEREO = True
            
            # ============================================================================
            # ПАРАЛЛЕЛЬНЫЙ ЗАПУСК ВЫЧИСЛЕНИЙ ГЛУБИНЫ РАЗНЫМИ МЕТОДАМИ
            # ============================================================================
            
            # ------------------------------------------------------------------------
            # ЗАПУСК СТЕРЕО В ОТДЕЛЬНОМ ПОТОКЕ (ЕСЛИ АКТИВЕН)
            # ------------------------------------------------------------------------
            
            # Инициализация переменной для хранения объекта будущего результата
            stereo_future = None
            # Замер времени начала вычислений стерео
            stereo_start = time.time()
            
            # Проверка флага активации стерео и наличия валидных изображений
            if USE_STEREO:
                # Отправка задачи в пул потоков:
                # 1. Функция: create_depth_map_stereo_scaled
                # 2. Аргументы: копии изображений (для потокобезопасности), параметры алгоритма
                stereo_future = executor.submit(
                    create_depth_map_stereo_scaled,    # Целевая функция
                    processed_left.copy(),             # Копия левого изображения (защита от гонки данных)
                    processed_right.copy(),            # Копия правого изображения
                    MIN_DISP_BASE,                     # Минимальная диспаритет
                    num_disp_scaled,                   # Максимальная диспаритет (масштабированная)
                    window_size_scaled                 # Размер блока сравнения (масштабированный)
                )
            
            # ------------------------------------------------------------------------
            # ЗАПУСК MiDaS В ОТДЕЛЬНОМ ПОТОКЕ (ЕСЛИ АКТИВЕН)
            # ------------------------------------------------------------------------
            
            # Инициализация переменной для хранения объекта будущего результата MiDaS
            midas_future = None
            # Замер времени начала вычислений MiDaS
            midas_start = time.time()
            
            # Проверка флагов активации и наличия инициализированного оценщика
            if USE_MIDAS and midas_estimator is not None:
                # ВЫБОР ИСТОЧНИКА ИЗОБРАЖЕНИЯ ПРИ ПЕРЕКРЫТИИ:
                # При перекрытии левой камеры используем правую, и наоборот
                # Это критически важно для продолжения работы в резервном режиме
                if occlusion_state == 'left':
                    source_frame = processed_right  # Используем правую камеру
                else:
                    source_frame = processed_left   # Используем левую камеру (стандартный режим или перекрытие правой)
                
                # Отправка задачи в пул потоков:
                # Вызов объекта midas_estimator как функции (__call__) с копией кадра
                midas_future = executor.submit(
                    midas_estimator,           # Объект оценщика (вызывается как функция)
                    source_frame.copy()        # Копия кадра для потокобезопасности
                )
            
            # ------------------------------------------------------------------------
            # ВЫЧИСЛЕНИЕ ОПТИЧЕСКОГО ПОТОКА В ГЛАВНОМ ПОТОКЕ
            # ------------------------------------------------------------------------
            
            # Замер времени начала вычислений оптического потока
            flow_start = time.time()
            # Инициализация переменных результата
            flow_depth_raw = None      # Сырая карта глубины от оптического потока
            camera_moving = False      # Флаг текущего движения камеры
            
            # Проверка флага активации оптического потока
            if USE_OPTICAL_FLOW:
                # ВЫБОР ИСТОЧНИКА ИЗОБРАЖЕНИЯ ПРИ ПЕРЕКРЫТИИ (аналогично MiDaS):
                if occlusion_state == 'left':
                    source_frame = processed_right
                else:
                    source_frame = processed_left
                
                # ВЫЗОВ ОЦЕНЩИКА ОПТИЧЕСКОГО ПОТОКА:
                # Возвращает карту глубины или None если камера статична/ошибка
                flow_depth_raw = flow_estimator(source_frame)
                # Получение флага движения камеры из внутреннего состояния оценщика
                camera_moving = flow_estimator.camera_moving
            
            # ЗАМЕР ВРЕМЕНИ ВЫЧИСЛЕНИЯ ОПТИЧЕСКОГО ПОТОКА
            flow_time = time.time() - flow_start
            # НАКОПЛЕНИЕ ВРЕМЕНИ ДЛЯ СТАТИСТИКИ
            flow_time_sum += flow_time
            
            # ============================================================================
            # ПОЛУЧЕНИЕ РАЗМЕРА ИЗОБРАЖЕНИЯ ДЛЯ СОЗДАНИЯ РЕЗЕРВНЫХ МАССИВОВ
            # ============================================================================
            
            # Получение высоты и ширины обработанных изображений для создания пустых массивов при ошибках
            h, w = processed_left.shape[:2]
            
            # ============================================================================
            # ПОЛУЧЕНИЕ РЕЗУЛЬТАТОВ СТЕРЕО С ОБРАБОТКОЙ ОШИБОК
            # ============================================================================
            
            # Проверка необходимости получения результатов стерео
            if USE_STEREO and stereo_future is not None:
                try:
                    # ПОЛУЧЕНИЕ РЕЗУЛЬТАТОВ С ТАЙМАУТОМ 0.5 СЕКУНДЫ:
                    # Возвращает кортеж: (нормализованная_диспаритет, сырая_диспаритет, цветовая_карта, достоверность)
                    stereo_depth, stereo_disparity_raw, stereo_colormap, stereo_conf = stereo_future.result(timeout=0.5)
                    
                    # ЗАМЕР ФАКТИЧЕСКОГО ВРЕМЕНИ ВЫЧИСЛЕНИЯ
                    stereo_time = time.time() - stereo_start
                    # НАКОПЛЕНИЕ ВРЕМЕНИ ДЛЯ СТАТИСТИКИ
                    stereo_time_sum += stereo_time
                    
                except Exception as e:
                    # ОБРАБОТКА ЛЮБЫХ ИСКЛЮЧЕНИЙ (таймаут, ошибка вычислений и т.д.):
                    print(f"[ОШИБКА СТЕРЕО] {e}")
                    import traceback
                    traceback.print_exc()  # Вывод полного стека вызовов для диагностики
                    
                    # СОЗДАНИЕ ПУСТЫХ МАССИВОВ ДЛЯ ПРОДОЛЖЕНИЯ РАБОТЫ СИСТЕМЫ:
                    stereo_depth = np.zeros((h, w), dtype=np.float32)          # Нулевая карта диспаритета
                    stereo_disparity_raw = np.zeros((h, w), dtype=np.float32)  # Нулевая сырая диспаритет
                    stereo_colormap = cv2.applyColorMap(                       # Черная цветовая карта
                        np.zeros((h, w), dtype=np.uint8), 
                        cv2.COLORMAP_JET
                    )
                    stereo_conf = np.zeros((h, w), dtype=np.float32)           # Нулевая достоверность
                    
                    # Обнуление времени вычисления для статистики
                    stereo_time = 0.0
                    stereo_time_sum += stereo_time
            
            else:
                # СЛУЧАЙ: СТЕРЕО ОТКЛЮЧЕНО ИЛИ НЕДОСТУПНО:
                # Создание пустых массивов для унификации интерфейса обработки
                stereo_depth = np.zeros((h, w), dtype=np.float32)
                stereo_disparity_raw = np.zeros((h, w), dtype=np.float32)
                stereo_colormap = cv2.applyColorMap(
                    np.zeros((h, w), dtype=np.uint8), 
                    cv2.COLORMAP_JET
                )
                stereo_conf = np.zeros((h, w), dtype=np.float32)
                stereo_time = 0.0
                stereo_time_sum += stereo_time
            
            # ============================================================================
            # ПОЛУЧЕНИЕ И ОБРАБОТКА РЕЗУЛЬТАТОВ MiDaS
            # ============================================================================
            
            # Инициализация переменных для результатов калибровки MiDaS
            midas_depth_calibrated = None  # Откалиброванная карта глубины MiDaS
            midas_conf = None              # Карта достоверности MiDaS
            
            # Проверка наличия будущего результата MiDaS
            if midas_future is not None:
                try:
                    # ПОЛУЧЕНИЕ РЕЗУЛЬТАТОВ С ТАЙМАУТОМ 0.5 СЕКУНДЫ:
                    # Возвращает кортеж: (сырая_глубина, достоверность) или (None, None) при ошибке
                    midas_raw, midas_conf = midas_future.result(timeout=0.5)
                    
                    # ЗАМЕР ВРЕМЕНИ ВЫЧИСЛЕНИЯ И НАКОПЛЕНИЕ ДЛЯ СТАТИСТИКИ
                    midas_time = time.time() - midas_start
                    midas_time_sum += midas_time
                    
                    # ПРОВЕРКА УСПЕШНОСТИ ПОЛУЧЕНИЯ СЫРОЙ КАРТЫ ГЛУБИНЫ
                    if midas_raw is not None:
                        # ------------------------------------------------------------------------
                        # КАЛИБРОВКА ДИАПАЗОНА MiDaS К ДИАПАЗОНУ СТЕРЕО
                        # ------------------------------------------------------------------------
                        
                        # СЛУЧАЙ 1: СТЕРЕО ДОСТУПНО — ТОЧНАЯ КАЛИБРОВКА ПО ОПОРНЫМ ТОЧКАМ:
                        if USE_STEREO and stereo_depth is not None and stereo_conf is not None:
                            # ВАЖНО: ИСПОЛЬЗУЕМ СТЕРЕОДИСПАРИТЕТ В ДИАПАЗОНЕ [0,255] (stereo_depth),
                            # А НЕ СЫРУЮ ДИСПАРИТЕТ В ПИКСЕЛЯХ (stereo_disparity_raw)!
                            # Это критически важно для корректной калибровки диапазонов
                            midas_depth_calibrated = calibrate_midas_to_stereo(
                                midas_raw,        # Сырая карта глубины MiDaS
                                stereo_depth,     # Нормализованная стереодиспаритет [0,255]
                                stereo_conf       # Карта достоверности стерео
                            )
                        
                        # СЛУЧАЙ 2: СТЕРЕО НЕДОСТУПНО — ПРОСТАЯ НОРМАЛИЗАЦИЯ В [0, 255]:
                        else:
                            # Вычисление минимального и максимального значений
                            min_val = np.min(midas_raw)
                            max_val = np.max(midas_raw)
                            
                            # Линейная нормализация с защитой от деления на ноль
                            if max_val > min_val + 1e-6:
                                midas_normalized = (midas_raw - min_val) / (max_val - min_val + 1e-8)
                            else:
                                midas_normalized = np.zeros_like(midas_raw)  # Однородное изображение
                        
                            # Преобразование в диапазон [0, 255] и ограничение границ
                            midas_depth_calibrated = np.clip(midas_normalized * 255.0, 0, 255).astype(np.float32)
                        
                        # ------------------------------------------------------------------------
                        # ДОПОЛНИТЕЛЬНОЕ СГЛАЖИВАНИЕ КАРТЫ ГЛУБИНЫ MiDaS
                        # ------------------------------------------------------------------------
                        
                        # Проверка наличия данных и адекватности диапазона значений
                        if (midas_depth_calibrated is not None and 
                            np.max(midas_depth_calibrated) > 10.0):
                            # Применение билиатерального фильтра для сглаживания без размытия границ
                            midas_depth_calibrated = cv2.bilateralFilter(
                                midas_depth_calibrated.astype(np.float32),  # Входной массив
                                5,   # Диаметр окна фильтрации
                                50,  # Сигма цветового пространства
                                50   # Сигма координатного пространства
                            )
                
                except Exception as e:
                    # ОБРАБОТКА ОШИБОК MiDaS С ДЕТАЛЬНЫМ ЛОГИРОВАНИЕМ
                    print(f"[ОШИБКА MiDaS] {e}")
                    import traceback
                    traceback.print_exc()
                    # Обнуление времени вычисления для статистики
                    midas_time = 0.0
                    midas_time_sum += midas_time
            
            # ============================================================================
            # НОРМАЛИЗАЦИЯ КАРТЫ ГЛУБИНЫ ОПТИЧЕСКОГО ПОТОКА
            # ============================================================================
            
            # Инициализация переменной для нормализованной карты
            flow_depth_normalized = None
            
            # Проверка наличия сырых данных оптического потока
            if flow_depth_raw is not None:
                # СЛУЧАЙ 1: СТЕРЕО ДОСТУПНО — НОРМАЛИЗАЦИЯ К ДИАПАЗОНУ СТЕРЕОДИСПАРИТЕТА
                if USE_STEREO and stereo_disparity_raw is not None:
                    flow_depth_normalized = normalize_to_stereo_range(
                        flow_depth_raw,        # Сырая карта глубины от потока
                        stereo_disparity_raw   # Сырая стереодиспаритет в пикселях
                    )
                    # ИНВЕРСИЯ ЗНАЧЕНИЙ (ближе = больше значение):
                    # Оптический поток даёт обратную зависимость (ближе = больше остаток),
                    # поэтому инвертируем для согласования со стерео
                    if flow_depth_normalized is not None:
                        flow_depth_normalized = 255.0 - flow_depth_normalized
                
                # СЛУЧАЙ 2: СТЕРЕО НЕДОСТУПНО — ПРОСТАЯ НОРМАЛИЗАЦИЯ В [0, 255]
                else:
                    # Масштабирование в диапазон [0, 255] с ограничением границ
                    flow_depth_normalized = np.clip(flow_depth_raw * 255.0, 0, 255).astype(np.float32)
                    # Инверсия для согласования направления глубины
                    flow_depth_normalized = 255.0 - flow_depth_normalized
            
            # ============================================================================
            # АДАПТИВНЫЙ ФЬЮЖН КАРТ ГЛУБИНЫ
            # ============================================================================
            
            # Инициализация переменных для итоговой карты глубины
            fused_depth_uint8 = None   # Итоговая карта глубины в формате uint8 [0,255]
            fused_colormap = None      # Цветовая карта для визуализации
            
            # Проверка флага визуализации фьюжн-карты
            if show_fused_depth:
                # ВЫЗОВ ФУНКЦИИ АДАПТИВНОГО ФЬЮЖНА:
                fused_result = fuse_depth_maps(
                    # Параметры стерео:
                    stereo_depth, 
                    stereo_conf,
                    # Параметры MiDaS:
                    midas_depth_calibrated, 
                    midas_conf,
                    # Параметры оптического потока:
                    flow_depth_normalized, 
                    camera_moving,
                    # Флаги активации методов:
                    use_stereo=USE_STEREO,
                    use_midas=USE_MIDAS,
                    use_flow=USE_OPTICAL_FLOW
                )
                
                # ПРОВЕРКА УСПЕШНОСТИ ФЬЮЖНА И ИЗВЛЕЧЕНИЕ РЕЗУЛЬТАТОВ
                if fused_result is not None:
                    fused_depth_uint8, fused_colormap = fused_result
            
            # ============================================================================
            # ПОДГОТОВКА ИЗОБРАЖЕНИЙ ДЛЯ ВИЗУАЛИЗАЦИИ
            # ============================================================================
            
            # МАСШТАБИРОВАНИЕ ИСХОДНЫХ КАДРОВ ДО РАЗМЕРА ОТОБРАЖЕНИЯ:
            display_left = cv2.resize(
                frame_left, 
                DISPLAY_SIZE, 
                interpolation=cv2.INTER_LINEAR
            )
            display_right = cv2.resize(
                frame_right, 
                DISPLAY_SIZE, 
                interpolation=cv2.INTER_LINEAR
            )
            
            # ============================================================================
            # ФОРМИРОВАНИЕ СТРОКИ СТАТУСА ДЛЯ ОТОБРАЖЕНИЯ НА ИЗОБРАЖЕНИИ
            # ============================================================================
            
            # Инициализация списка компонентов статуса
            status_parts = []
            
            # ДОБАВЛЕНИЕ ИНДИКАТОРА СОСТОЯНИЯ ПЕРЕКРЫТИЯ:
            if occlusion_state == 'none':
                status_parts.append("✓ Обе камеры свободны")
            elif occlusion_state == 'left':
                status_parts.append("⚠ ЛЕВАЯ ПЕРЕКРЫТА")
            elif occlusion_state == 'right':
                status_parts.append("⚠ ПРАВАЯ ПЕРЕКРЫТА")
            else:  # 'both'
                status_parts.append("ⓘ Обе перекрыты")
            
            # ДОБАВЛЕНИЕ ИНДИКАТОРА СОСТОЯНИЯ СТЕРЕО:
            if USE_STEREO:
                status_parts.append("Stereo:ON")
            else:
                status_parts.append("Stereo:OFF")
            
            # ДОБАВЛЕНИЕ ИНДИКАТОРА СОСТОЯНИЯ MiDaS (если активен):
            if USE_MIDAS:
                status_parts.append("MiDaS:ON")
            
            # ДОБАВЛЕНИЕ ИНДИКАТОРА ДВИЖЕНИЯ КАМЕРЫ (если активен поток и камера движется):
            if USE_OPTICAL_FLOW and camera_moving:
                status_parts.append("Flow:MOVING")
            
            # ОБЪЕДИНЕНИЕ КОМПОНЕНТОВ В ЕДИНУЮ СТРОКУ С РАЗДЕЛИТЕЛЕМ
            status_text = " | ".join(status_parts)
            
            # ============================================================================
            # НАЛОЖЕНИЕ ТЕКСТОВОЙ ИНФОРМАЦИИ НА ИЗОБРАЖЕНИЯ КАМЕР
            # ============================================================================
            
            # НАЛОЖЕНИЕ СТАТУСА НА ЛЕВОЕ ИЗОБРАЖЕНИЕ:
            cv2.putText(
                display_left,                          # Целевое изображение
                f"Левая | {status_text} | Scale:{PROCESSING_SCALE:.2f}x",  # Текст
                (10, 30),                              # Позиция (x, y)
                cv2.FONT_HERSHEY_COMPLEX,              # Шрифт
                0.6,                                   # Масштаб шрифта
                (0, 255, 0),                           # Цвет в формате BGR (зеленый)
                2                                      # Толщина линий
            )
            
            # НАЛОЖЕНИЕ ПРОСТОГО ИНДИКАТОРА НА ПРАВОЕ ИЗОБРАЖЕНИЕ:
            cv2.putText(
                display_right,
                "Правая",
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            # ============================================================================
            # ВИЗУАЛЬНАЯ ИНДИКАЦИЯ ПЕРЕКРЫТИЯ КАМЕР (КРАСНАЯ РАМКА)
            # ============================================================================
            
            # ИНДИКАЦИЯ ПЕРЕКРЫТИЯ ЛЕВОЙ КАМЕРЫ:
            if occlusion_state == 'left':
                # Рисование красной рамки по периметру изображения
                cv2.rectangle(
                    display_left,
                    (0, 0),                                      # Верхний левый угол
                    (DISPLAY_SIZE[0]-1, DISPLAY_SIZE[1]-1),      # Нижний правый угол
                    (0, 0, 255),                                 # Цвет BGR (красный)
                    3                                                # Толщина линии
                )
                # Наложение текстового предупреждения внизу изображения
                cv2.putText(
                    display_left,
                    "ПЕРЕКРЫТО!",
                    (10, DISPLAY_SIZE[1]-20),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6,
                    (0, 0, 255),  # Красный цвет
                    2
                )
            
            # ИНДИКАЦИЯ ПЕРЕКРЫТИЯ ПРАВОЙ КАМЕРЫ:
            elif occlusion_state == 'right':
                cv2.rectangle(
                    display_right,
                    (0, 0),
                    (DISPLAY_SIZE[0]-1, DISPLAY_SIZE[1]-1),
                    (0, 0, 255),
                    3
                )
                cv2.putText(
                    display_right,
                    "ПЕРЕКРЫТО!",
                    (10, DISPLAY_SIZE[1]-20),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
            
            # ============================================================================
            # СОЗДАНИЕ КОМБИНИРОВАННОГО ИЗОБРАЖЕНИЯ ДЛЯ ОКНА "Stereo cameras"
            # ============================================================================
            
            # Горизонтальная стыковка левого и правого изображений
            combined = np.hstack((display_left, display_right))
            
            # macOS FIX: УСТАНОВКА ОБРАБОТЧИКА КЛИКА ДЛЯ ПЕРВОГО КАДРА
            # Предотвращает потерю фокуса ввода после перетаскивания окна
            if sys.platform == 'darwin' and frame_counter == 1:
                cv2.namedWindow('Stereo cameras', cv2.WINDOW_NORMAL)
                # Пустой обработчик клика (игнорирует все события мыши)
                cv2.setMouseCallback('Stereo cameras', lambda *args: None)
            
            # ============================================================================
            # НАЛОЖЕНИЕ ИНСТРУКЦИИ ПО УПРАВЛЕНИЮ НА КОМБИНИРОВАННОЕ ИЗОБРАЖЕНИЕ
            # ============================================================================
            
            # ФОРМИРОВАНИЕ МНОГОСТРОЧНОЙ ИНСТРУКЦИИ:
            instruction_text = (
                "'q'-выход | 'c'-сохранить | 'v'-ректиф | "
                "'z'-стереометод | 'm'-MiDaS | 'o'-поток | "
                "'1/2/3/4'-карты | '5'-панель | 'p'-сброс | "
                "'+/- масштаб' | настройки:w/s/e/d/r/f/t/g/y/h"
            )
            
            # ВЫЗОВ ФУНКЦИИ МНОГОСТРОЧНОГО ТЕКСТА С АВТОМАТИЧЕСКИМ ПЕРЕНОСОМ СЛОВ:
            put_multiline_text_anywhere(
                combined,           # Целевое изображение
                instruction_text,   # Текст для отображения
                position='bottom',  # Позиционирование внизу изображения
                font_face=cv2.FONT_HERSHEY_COMPLEX,
                font_scale=0.6,
                color=(255, 128, 0),  # Оранжевый цвет в формате BGR
                thickness=2,
                margin=20,            # Отступ от краев
                line_spacing=30       # Расстояние между строками
            )
            
            # ОТОБРАЖЕНИЕ КОМБИНИРОВАННОГО ИЗОБРАЖЕНИЯ В ОКНЕ
            cv2.imshow('Stereo cameras', combined)
            
            # ============================================================================
            # ВИЗУАЛИЗАЦИЯ КАРТ ГЛУБИНЫ (ПО ФЛАГАМ)
            # ============================================================================
            
            # ВИЗУАЛИЗАЦИЯ СТЕРЕОГЛУБИНЫ:
            if show_depth:
                # macOS FIX: установка обработчика клика для окна
                if sys.platform == 'darwin' and frame_counter == 1:
                    cv2.namedWindow('Depth map', cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback('Depth map', lambda *args: None)
                
                # Отображение цветовой карты стереоглубины в масштабированном размере
                cv2.imshow(
                    'Depth map', 
                    cv2.resize(stereo_colormap, DISPLAY_SIZE)
                )
            
            # ВИЗУАЛИЗАЦИЯ ГЛУБИНЫ MiDaS:
            if show_midas_depth and midas_depth_calibrated is not None:
                # Преобразование в диапазон [0, 255] и тип uint8
                midas_uint8 = np.clip(midas_depth_calibrated, 0, 255).astype(np.uint8)
                # Создание цветовой карты
                midas_colormap = cv2.applyColorMap(midas_uint8, cv2.COLORMAP_JET)
                # Наложение заголовка
                cv2.putText(
                    midas_colormap,
                    "MiDaS Depth",
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (255, 255, 255),  # Белый цвет
                    2
                )
                # macOS FIX: установка обработчика клика
                if sys.platform == 'darwin' and frame_counter == 1:
                    cv2.namedWindow('MiDaS Depth', cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback('MiDaS Depth', lambda *args: None)
                # Отображение
                cv2.imshow(
                    'MiDaS Depth', 
                    cv2.resize(midas_colormap, DISPLAY_SIZE)
                )
            
            # ВИЗУАЛИЗАЦИЯ ГЛУБИНЫ ОПТИЧЕСКОГО ПОТОКА:
            if show_flow_depth and flow_depth_normalized is not None:
                # Преобразование в диапазон [0, 255] и тип uint8
                flow_uint8 = np.clip(flow_depth_normalized, 0, 255).astype(np.uint8)
                # Создание цветовой карты
                flow_colormap = cv2.applyColorMap(flow_uint8, cv2.COLORMAP_JET)
                # Определение статуса движения
                motion_status = "ДВИЖЕНИЕ" if camera_moving else "СТАТИКА"
                # Наложение заголовка с индикацией движения
                cv2.putText(
                    flow_colormap,
                    f"Flow ({motion_status})",
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                # macOS FIX: установка обработчика клика
                if sys.platform == 'darwin' and frame_counter == 1:
                    cv2.namedWindow('Flow Depth', cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback('Flow Depth', lambda *args: None)
                # Отображение
                cv2.imshow(
                    'Flow Depth', 
                    cv2.resize(flow_colormap, DISPLAY_SIZE)
                )
            
            # ВИЗУАЛИЗАЦИЯ ИТОГОВОЙ ФЬЮЖН-КАРТЫ ГЛУБИНЫ:
            if show_fused_depth and fused_colormap is not None:
                # ДОПОЛНИТЕЛЬНАЯ ИНДИКАЦИЯ РЕЗЕРВНОГО РЕЖИМА ПРИ ПЕРЕКРЫТИИ:
                if occlusion_state in ['left', 'right']:
                    occlusion_text = (
                        f"РЕЗЕРВНЫЙ РЕЖИМ: {'правая' if occlusion_state == 'left' else 'левая'} камера"
                    )
                    cv2.putText(
                        fused_colormap,
                        occlusion_text,
                        (10, 85),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (0, 165, 255),  # Оранжевый цвет
                        2
                    )
                # macOS FIX: установка обработчика клика
                if sys.platform == 'darwin' and frame_counter == 1:
                    cv2.namedWindow('FUSED Depth', cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback('FUSED Depth', lambda *args: None)
                # Отображение
                cv2.imshow(
                    'FUSED Depth', 
                    cv2.resize(fused_colormap, DISPLAY_SIZE)
                )
            
            # ОТОБРАЖЕНИЕ ПАНЕЛИ ПАРАМЕТРОВ ФЬЮЖНА (ЕСЛИ АКТИВНА):
            if show_fusion_panel:
                # ВЫЗОВ ФУНКЦИИ ОТОБРАЖЕНИЯ ПАНЕЛИ С ТЕКУЩИМИ ПАРАМЕТРАМИ:
                # Передача текущего состояния перекрытия и оценок для визуализации
                display_fusion_params_panel(
                    occlusion_state,  # Текущее состояние перекрытия
                    left_score,       # Оценка перекрытия левой камеры
                    right_score       # Оценка перекрытия правой камеры
                )
            
            # ============================================================================
            # ПЕРИОДИЧЕСКИЙ ВЫВОД СТАТИСТИКИ ПРОИЗВОДИТЕЛЬНОСТИ
            # ============================================================================
            
            # ПРОВЕРКА УСЛОВИЯ ВЫВОДА СТАТИСТИКИ (каждые 30 обработанных кадров):
            if total_frames % 30 == 0:
                # ВЫЧИСЛЕНИЕ ВРЕМЕНИ, ПРОШЕДШЕГО С ПОСЛЕДНЕГО ВЫВОДА
                elapsed = time.time() - last_stats_time
                # РАСЧЁТ СРЕДНЕЙ ЧАСТОТЫ КАДРОВ (FPS)
                fps = 30.0 / elapsed if elapsed > 0 else 0.0
                
                # ВЫВОД ЗАГОЛОВКА СТАТИСТИКИ С ИНФОРМАЦИЕЙ О ПАРАМЕТРАХ
                print(f"\n[СТАТИСТИКА ЗА 30 ОБРАБОТАННЫХ КАДРОВ | Масштаб: {PROCESSING_SCALE:.2f}x | Пропуск: каждый {SKIP_FRAMES}-й]")
                print(f"  Частота обработки: {fps:.1f} кадров/сек")
                print(f"  Среднее время стерео: {stereo_time_sum/30*1000:.1f} мс (вкл: {USE_STEREO})")
                # Условный вывод статистики MiDaS (только если активен)
                if USE_MIDAS:
                    print(f"  Среднее время MiDaS: {midas_time_sum/30*1000:.1f} мс")
                print(f"  Среднее время потока: {flow_time_sum/30*1000:.1f} мс")
                # Вывод статуса перекрытия камер с числовыми оценками
                print(f"  Статус камер: {occlusion_state.upper()} (L_score={left_score:.2f}, R_score={right_score:.2f})")
                
                # СБРОС НАКОПЛЕННЫХ ЗНАЧЕНИЙ ВРЕМЕНИ ДЛЯ СЛЕДУЮЩЕГО ПЕРИОДА
                stereo_time_sum = 0.0
                midas_time_sum = 0.0
                flow_time_sum = 0.0
                # ОБНОВЛЕНИЕ ВРЕМЕНИ ПОСЛЕДНЕГО ВЫВОДА СТАТИСТИКИ
                last_stats_time = time.time()
            
            # ============================================================================
            # ОЖИДАНИЕ И ОБРАБОТКА КЛАВИШ ВВОДА
            # ============================================================================
            
            # ОЖИДАНИЕ НАЖАТИЯ КЛАВИШИ С УВЕЛИЧЕННОЙ ЗАДЕРЖКОЙ ДЛЯ macOS (10 мс):
            key = cv2.waitKey(10) & 0xFF  # Маскирование для получения младшего байта
            
            # ------------------------------------------------------------------------
            # ОБРАБОТКА КОМАНД УПРАВЛЕНИЯ
            # ------------------------------------------------------------------------
            
            # ВЫХОД ИЗ ПРОГРАММЫ:
            if key == ord('q'):
                print("\n✅ Запрошен выход из программы пользователем")
                break
            
            # ПЕРЕКЛЮЧЕНИЕ СТЕРЕОРЕКТИФИКАЦИИ:
            elif key == ord('v'):
                use_stereo_rectify = not use_stereo_rectify
                status = "ВКЛ ✅" if use_stereo_rectify else "ВЫКЛ ⚠️ (геометрия НАРУШЕНА!)"
                print(f"Стереоректификация: {status}")
            
            # ПЕРЕКЛЮЧЕНИЕ МЕТОДА СТЕРЕОЗРЕНИЯ (остановка вычислений):
            elif key == ord('z'):
                USE_STEREO = not USE_STEREO
                status = "ВКЛ ✅" if USE_STEREO else "ВЫКЛ ⚠️ (вычисления отключены)"
                print(f"Метод стереозрения: {status}")
            
            # ПЕРЕКЛЮЧЕНИЕ ВИЗУАЛИЗАЦИИ СТЕРЕОГЛУБИНЫ:
            elif key == ord('1'):
                show_depth = not show_depth
                if show_depth:  # При включении — отключаем другие визуализации
                    show_midas_depth = False
                    show_flow_depth = False
                    show_fused_depth = False
                print(f"Визуализация стереоглубины: {'ВКЛ' if show_depth else 'ВЫКЛ'}")
            
            # ПЕРЕКЛЮЧЕНИЕ ВИЗУАЛИЗАЦИИ ГЛУБИНЫ MiDaS:
            elif key == ord('2'):
                show_midas_depth = not show_midas_depth
                if show_midas_depth:
                    show_depth = False
                    show_flow_depth = False
                    show_fused_depth = False
                print(f"MiDaS карта глубины: {'ВКЛ' if show_midas_depth else 'ВЫКЛ'}")
            
            # ПЕРЕКЛЮЧЕНИЕ ВИЗУАЛИЗАЦИИ ГЛУБИНЫ ОПТИЧЕСКОГО ПОТОКА:
            elif key == ord('3'):
                show_flow_depth = not show_flow_depth
                if show_flow_depth:
                    show_depth = False
                    show_midas_depth = False
                    show_fused_depth = False
                print(f"Карта оптического потока: {'ВКЛ' if show_flow_depth else 'ВЫКЛ'}")
            
            # ПЕРЕКЛЮЧЕНИЕ ВИЗУАЛИЗАЦИИ ИТОГОВОЙ ФЬЮЖН-КАРТЫ:
            elif key == ord('4'):
                show_fused_depth = not show_fused_depth
                if show_fused_depth:
                    show_depth = False
                    show_midas_depth = False
                    show_flow_depth = False
                print(f"Итоговая фьюжн-карта: {'ВКЛ' if show_fused_depth else 'ВЫКЛ'}")
            
            # ПЕРЕКЛЮЧЕНИЕ ПАНЕЛИ ПАРАМЕТРОВ ФЬЮЖНА:
            elif key == ord('5'):
                show_fusion_panel = not show_fusion_panel
                if not show_fusion_panel:
                    # Закрытие окна панели при отключении
                    cv2.destroyWindow('Fusion Parameters')
                print(f"Панель параметров фьюжна: {'ВКЛ' if show_fusion_panel else 'ВЫКЛ'}")
            
            # ПЕРЕКЛЮЧЕНИЕ МЕТОДА MiDaS (только если доступен):
            elif key == ord('m') and TORCH_AVAILABLE:
                USE_MIDAS = not USE_MIDAS
                print(f"MiDaS метод: {'ВКЛ' if USE_MIDAS else 'ВЫКЛ'}")
            
            # ПЕРЕКЛЮЧЕНИЕ МЕТОДА ОПТИЧЕСКОГО ПОТОКА:
            elif key == ord('o'):
                USE_OPTICAL_FLOW = not USE_OPTICAL_FLOW
                print(f"Оптический поток: {'ВКЛ' if USE_OPTICAL_FLOW else 'ВЫКЛ'}")
            
            # СБРОС ПАРАМЕТРОВ ФЬЮЖНА К ЗНАЧЕНИЯМ ПО УМОЛЧАНИЮ:
            elif key == ord('p'):
                fusion_params['stereo_weight'] = FUSION_WEIGHTS['stereo_base']
                fusion_params['midas_fill_weight'] = FUSION_WEIGHTS['midas_max_fill']
                fusion_params['flow_fill_weight'] = FUSION_WEIGHTS['flow_max_fill']
                fusion_params['stereo_conf_threshold'] = FUSION_THRESHOLDS['stereo_low_conf']
                fusion_params['flow_hole_threshold'] = FUSION_THRESHOLDS['flow_hole_threshold']
                print("✅ Параметры фьюжна сброшены к значениям по умолчанию")
            
            # УВЕЛИЧЕНИЕ ВЕСА СТЕРЕО:
            elif key == ord('w'):
                fusion_params['stereo_weight'] = min(1.5, fusion_params['stereo_weight'] + 0.1)
                print(f"Вес стерео увеличен: {fusion_params['stereo_weight']:.2f}")
            
            # УМЕНЬШЕНИЕ ВЕСА СТЕРЕО:
            elif key == ord('s'):
                fusion_params['stereo_weight'] = max(0.1, fusion_params['stereo_weight'] - 0.1)
                print(f"Вес стерео уменьшен: {fusion_params['stereo_weight']:.2f}")
            
            # УВЕЛИЧЕНИЕ ВЕСА ЗАПОЛНЕНИЯ MiDaS:
            elif key == ord('e'):
                fusion_params['midas_fill_weight'] = min(1.0, fusion_params['midas_fill_weight'] + 0.1)
                print(f"Вес заполнения MiDaS увеличен: {fusion_params['midas_fill_weight']:.2f}")
            
            # УМЕНЬШЕНИЕ ВЕСА ЗАПОЛНЕНИЯ MiDaS:
            elif key == ord('d'):
                fusion_params['midas_fill_weight'] = max(0.0, fusion_params['midas_fill_weight'] - 0.1)
                print(f"Вес заполнения MiDaS уменьшен: {fusion_params['midas_fill_weight']:.2f}")
            
            # УВЕЛИЧЕНИЕ ВЕСА ЗАПОЛНЕНИЯ ПОТОКОМ:
            elif key == ord('r'):
                fusion_params['flow_fill_weight'] = min(1.0, fusion_params['flow_fill_weight'] + 0.1)
                print(f"Вес заполнения потоком увеличен: {fusion_params['flow_fill_weight']:.2f}")
            
            # УМЕНЬШЕНИЕ ВЕСА ЗАПОЛНЕНИЯ ПОТОКОМ:
            elif key == ord('f'):
                fusion_params['flow_fill_weight'] = max(0.0, fusion_params['flow_fill_weight'] - 0.1)
                print(f"Вес заполнения потоком уменьшен: {fusion_params['flow_fill_weight']:.2f}")
            
            # УВЕЛИЧЕНИЕ ПОРОГА ДОСТОВЕРНОСТИ СТЕРЕО:
            elif key == ord('t'):
                fusion_params['stereo_conf_threshold'] = min(0.95, fusion_params['stereo_conf_threshold'] + 0.05)
                print(f"Порог уверенности стерео увеличен: {fusion_params['stereo_conf_threshold']:.2f}")
            
            # УМЕНЬШЕНИЕ ПОРОГА ДОСТОВЕРНОСТИ СТЕРЕО:
            elif key == ord('g'):
                fusion_params['stereo_conf_threshold'] = max(0.1, fusion_params['stereo_conf_threshold'] - 0.05)
                print(f"Порог уверенности стерео уменьшен: {fusion_params['stereo_conf_threshold']:.2f}")
            
            # УВЕЛИЧЕНИЕ ПОРОГА "ДЫР" ДЛЯ ПОТОКА:
            elif key == ord('y'):
                fusion_params['flow_hole_threshold'] = min(50.0, fusion_params['flow_hole_threshold'] + 1.0)
                print(f"Порог 'дыр' для потока увеличен: {fusion_params['flow_hole_threshold']:.1f}")
            
            # УМЕНЬШЕНИЕ ПОРОГА "ДЫР" ДЛЯ ПОТОКА:
            elif key == ord('h'):
                fusion_params['flow_hole_threshold'] = max(1.0, fusion_params['flow_hole_threshold'] - 1.0)
                print(f"Порог 'дыр' для потока уменьшен: {fusion_params['flow_hole_threshold']:.1f}")
            
            # СОХРАНЕНИЕ ТЕКУЩИХ КАДРОВ И КАРТ ГЛУБИНЫ:
            elif key == ord('c'):
                # Генерация уникального временной метки на основе текущего времени
                timestamp = int(time.time() * 1000)
                # Создание директории для сохранения (если не существует)
                os.makedirs("depth_captures", exist_ok=True)
                
                # Сохранение исходных кадров с камер
                cv2.imwrite(f"depth_captures/left_{timestamp}.jpg", frame_left)
                cv2.imwrite(f"depth_captures/right_{timestamp}.jpg", frame_right)
                
                # Условное сохранение карт глубины (если они визуализируются и доступны)
                if show_depth and stereo_colormap is not None:
                    cv2.imwrite(
                        f"depth_captures/stereo_depth_{timestamp}.png",
                        cv2.resize(stereo_colormap, DISPLAY_SIZE)
                    )
                if show_midas_depth and midas_depth_calibrated is not None:
                    midas_uint8 = np.clip(midas_depth_calibrated, 0, 255).astype(np.uint8)
                    cv2.imwrite(
                        f"depth_captures/midas_depth_{timestamp}.png",
                        cv2.resize(cv2.applyColorMap(midas_uint8, cv2.COLORMAP_JET), DISPLAY_SIZE)
                    )
                if show_flow_depth and flow_depth_normalized is not None:
                    flow_uint8 = np.clip(flow_depth_normalized, 0, 255).astype(np.uint8)
                    cv2.imwrite(
                        f"depth_captures/flow_depth_{timestamp}.png",
                        cv2.resize(cv2.applyColorMap(flow_uint8, cv2.COLORMAP_JET), DISPLAY_SIZE)
                    )
                if show_fused_depth and fused_colormap is not None:
                    cv2.imwrite(
                        f"depth_captures/fused_depth_{timestamp}.png",
                        cv2.resize(fused_colormap, DISPLAY_SIZE)
                    )
                
                # Подтверждение сохранения пользователю
                print(f"✅ Кадры и карты глубины сохранены в папку depth_captures/ (timestamp={timestamp})")
            
            # УВЕЛИЧЕНИЕ МАСШТАБА ОБРАБОТКИ:
            elif key == ord('+') or key == ord('='):
                # Вычисление нового масштаба с ограничением максимума 1.0 (полное разрешение)
                new_scale = min(1.0, PROCESSING_SCALE + 0.1)
                # Проверка значимости изменения (избегаем перезагрузки при минимальных изменениях)
                if abs(new_scale - PROCESSING_SCALE) > 0.01:
                    PROCESSING_SCALE = new_scale
                    print(f"\n🔄 Масштаб обработки увеличен: {PROCESSING_SCALE:.2f}x")
                    print("   Перезагрузка калибровки с ректификацией...")
                    # Перезагрузка калибровки с новым масштабом
                    stereo_calib = load_stereo_calibration_with_scaling(scale_factor=PROCESSING_SCALE)
                    # Пересчет параметров стереоалгоритма под новый масштаб
                    num_disp_scaled = max(16, int(NUM_DISP_BASE * PROCESSING_SCALE) // 16 * 16)
                    window_size_scaled = max(5, int(WINDOW_SIZE_BASE * PROCESSING_SCALE))
                    if window_size_scaled % 2 == 0:
                        window_size_scaled += 1
                    print(f"   Новые параметры: numDisp={num_disp_scaled}, blockSize={window_size_scaled}")
            
            # УМЕНЬШЕНИЕ МАСШТАБА ОБРАБОТКИ:
            elif key == ord('-') or key == ord('_'):
                # Вычисление нового масштаба с ограничением минимума 0.3 (слишком низкий масштаб снижает качество)
                new_scale = max(0.3, PROCESSING_SCALE - 0.1)
                if abs(new_scale - PROCESSING_SCALE) > 0.01:
                    PROCESSING_SCALE = new_scale
                    print(f"\n🔄 Масштаб обработки уменьшен: {PROCESSING_SCALE:.2f}x")
                    print("   Перезагрузка калибровки с ректификацией...")
                    stereo_calib = load_stereo_calibration_with_scaling(scale_factor=PROCESSING_SCALE)
                    num_disp_scaled = max(16, int(NUM_DISP_BASE * PROCESSING_SCALE) // 16 * 16)
                    window_size_scaled = max(5, int(WINDOW_SIZE_BASE * PROCESSING_SCALE))
                    if window_size_scaled % 2 == 0:
                        window_size_scaled += 1
                    print(f"   Новые параметры: numDisp={num_disp_scaled}, blockSize={window_size_scaled}")
    
    # ============================================================================
    # ОБРАБОТКА ИСКЛЮЧЕНИЙ И КОРРЕКТНОЕ ЗАВЕРШЕНИЕ ПРОГРАММЫ
    # ============================================================================
    
    except KeyboardInterrupt:
        # Перехват прерывания через Ctrl+C
        print("\n⚠️  Прервано пользователем (Ctrl+C)")
    
    except Exception as e:
        # Перехват любых других исключений с детальным логированием
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()  # Вывод полного стека вызовов
    
    finally:
        # ============================================================================
        # КОРРЕКТНОЕ ОСВОБОЖДЕНИЕ ВСЕХ РЕСУРСОВ (ГАРАНТИРОВАННОЕ ВЫПОЛНЕНИЕ)
        # ============================================================================
        
        print("\n🔄 Завершение работы программы...")
        
        # ОСТАНОВКА ПУЛА ПОТОКОВ С ОЖИДАНИЕМ ЗАВЕРШЕНИЯ ВСЕХ ЗАДАЧ
        executor.shutdown(wait=True)
        print("✅ Пул потоков остановлен")
        
        # ОСВОБОЖДЕНИЕ РЕСУРСОВ КАМЕР (ЕСЛИ ОНИ БЫЛИ ИНИЦИАЛИЗИРОВАНЫ)
        if left_cap is not None and left_cap.isOpened():
            left_cap.release()
            print("✅ Левая камера закрыта")
        if right_cap is not None and right_cap.isOpened():
            right_cap.release()
            print("✅ Правая камера закрыта")
        
        # ЗАКРЫТИЕ ВСЕХ ОКОН OpenCV
        cv2.destroyAllWindows()
        print("✅ Все окна OpenCV закрыты")
        
        print("\n🏁 Программа завершена. Все ресурсы освобождены.")
        
if __name__ == "__main__":
    main()
    
    