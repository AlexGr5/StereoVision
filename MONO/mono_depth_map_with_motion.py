"""
Построение карты глубины на одной камере с помощью её перемещения.
Использует оптический поток для оценки параллакса и преобразования в карту глубины.
"""

import cv2
import numpy as np
import time


class CleanDisplayDepthEstimator:
    """
    Класс для оценки карты глубины на основе движения одной камеры.
    Использует оптический поток для выделения параллакса и оценки расстояния до объектов.
    """
    
    def __init__(self):
        """
        Инициализация класса CleanDisplayDepthEstimator.
        
        Создает видеозахват, настраивает параметры камеры и инициализирует
        переменные для отслеживания движения и стабилизации.
        """
        # Инициализация видеозахвата с камеры (индекс 0 - обычно встроенная камера)
        self.cap = cv2.VideoCapture(0)
        
        # Установка разрешения кадра: ширина 640 пикселей
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # Установка разрешения кадра: высота 480 пикселей
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Предыдущий кадр в оттенках серого (для расчета оптического потока)
        self.prev_gray = None
        # Предыдущий размер кадра
        self.prev_size = None
        
        # Накопленный оптический поток для внутренней стабилизации (не отображается)
        self.accumulated_flow = None
        
        # Минимальное количество точек-выбросов для валидации движения камеры
        self.min_inliers = 15
        
        # Время последнего детектированного движения камеры
        self.last_move_time = time.time()
        
        # Флаг движения камеры (True - камера движется, False - статична)
        self.camera_moving = False
        
        # Стабилизированная карта глубины для внутренних вычислений (не отображается)
        self.stable_depth = None
    
    def get_depth_map(self, frame):
        """
        Основная функция получения карты глубины из текущего кадра.
        
        Параметры:
        -----------
        frame : numpy.ndarray
            Текущий кадр в цветовом формате BGR
            
        Возвращает:
        -----------
        display_depth : numpy.ndarray или None
            Карта глубины для отображения или None, если недостаточно данных
        """
        # Преобразование кадра в оттенки серого для расчета оптического потока
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Инициализация при первом вызове функции
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_size = gray.shape[:2]
            # Инициализация матрицы накопленного потока (2 канала: dx, dy)
            self.accumulated_flow = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)
            return None
        
        # Проверка, что размеры совпадают
        if gray.shape != self.prev_gray.shape:
            # Если размеры не совпадают, сбросим предыдущий кадр
            self.prev_gray = gray.copy()
            self.prev_size = gray.shape[:2]
            self.accumulated_flow = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)
            self.stable_depth = None
            return None
        
        # Проверка, двигалась ли камера в последние 1.5 секунды
        current_time = time.time()
        self.camera_moving = (current_time - self.last_move_time) < 1.5
        
        try:
            # 1. Вычисление плотного оптического потока между кадрами
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,  # Исходные и целевое изображения
                0.5,  # Пирамидальное масштабирование (0.5 - каждый уровень в 2 раза меньше)
                3,    # Количество уровней пирамиды
                15,   # Размер окна для сглаживания
                3,    # Количество итераций на каждом уровне пирамиды
                5,    # Размер окна для вычисления стандартного отклонения
                1.2,  # Флаг расширения окна
                0     # Флаги алгоритма
            )
        except cv2.error as e:
            # Если возникает ошибка при вычислении оптического потока
            print(f"Ошибка оптического потока: {e}")
            self.prev_gray = gray.copy()
            return None
        
        # 2. Оценка эго-движения камеры (глобального смещения) через RANSAC
        ego_motion_valid, expected_flow = self.estimate_ego_motion(flow)
        
        if ego_motion_valid:
            # Обновление времени последнего валидного движения камеры
            self.last_move_time = current_time
            self.camera_moving = True
            
            # 3. Вычисление остаточного потока (параллакс - разница между общим и эго-движением)
            residual_flow_x = flow[..., 0] - expected_flow[..., 0]  # Разница по оси X
            residual_flow_y = flow[..., 1] - expected_flow[..., 1]  # Разница по оси Y
            
            # Вычисление величины остаточного потока (евклидова норма)
            residual_magnitude = np.sqrt(residual_flow_x**2 + residual_flow_y**2)
            
            # 4. Преобразование остаточного потока в карту глубины
            # Чем больше параллакс (движение объекта), тем он ближе (меньшая глубина)
            current_depth = 1.0 / (residual_magnitude + 0.5)  # +0.5 для избежания деления на 0
            
            # ВНУТРЕННЕЕ НАКОПЛЕНИЕ ДЛЯ СТАБИЛЬНОСТИ (не отображается!)
            if self.stable_depth is None:
                self.stable_depth = current_depth.copy()  # Инициализация при первом кадре
            else:
                # Экспоненциальное сглаживание: более агрессивное при движении (alpha=0.9),
                # более плавное в статике (alpha=0.99)
                alpha = 0.9 if self.camera_moving else 0.99
                self.stable_depth = alpha * self.stable_depth + (1 - alpha) * current_depth
            
            # 5. Возвращаем ТОЛЬКО ТЕКУЩУЮ КАРТУ без накопления для отображения
            display_depth = current_depth.copy()
            
            # 6. Сглаживание ТОЛЬКО текущего кадра с сохранением границ (edge-aware фильтр)
            display_depth = cv2.bilateralFilter(display_depth, 9, 75, 75)
            
            # Обновление предыдущего кадра для следующей итерации
            self.prev_gray = gray.copy()
            return display_depth
        
        # Если движение не валидно, но камера двигалась недавно - показываем стабильную карту
        if self.camera_moving and self.stable_depth is not None:
            # Возвращаем ТОЛЬКО текущую стабильную карту без изменений
            return cv2.bilateralFilter(self.stable_depth.copy(), 9, 75, 75)
        
        # Обновление предыдущего кадра при отсутствии валидного движения
        self.prev_gray = gray.copy()
        return None
    
    def estimate_ego_motion(self, flow):
        """
        Оценивает глобальное движение камеры (эго-движение) через RANSAC.
        
        Параметры:
        -----------
        flow : numpy.ndarray
            Матрица оптического потока размером (H, W, 2)
            
        Возвращает:
        -----------
        (success, expected_flow) : tuple
            success : bool
                Успешность оценки движения
            expected_flow : numpy.ndarray
                Ожидаемый поток для всего кадра на основе гомографии
        """
        # Получение размеров кадра
        h, w = flow.shape[:2]
        
        # Проверка минимального размера кадра
        if h < 50 or w < 50:
            return False, np.zeros_like(flow)
        
        # Создание равномерной сетки точек (шаг 16 пикселей)
        step = 16
        # Создание координатной сетки с шагом step, начиная с половины шага от края
        y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
        # Преобразование в массив точек (координаты x, y)
        points = np.vstack((x_coords, y_coords)).T.astype(np.float32)
        
        # Проверка, что точки находятся в пределах изображения
        if len(points) == 0:
            return False, np.zeros_like(flow)
        
        # Выборка векторов потока в точках сетки
        try:
            flow_vectors = flow[y_coords.astype(int), x_coords.astype(int)]
        except IndexError as e:
            print(f"Ошибка индексации потока: {e}")
            return False, np.zeros_like(flow)
        
        # Вычисление ожидаемых позиций точек в следующем кадре
        next_points = points + flow_vectors.reshape(-1, 2)
        
        # Проверка достаточности точек для RANSAC
        if len(points) < self.min_inliers * 2:
            return False, np.zeros_like(flow)
        
        try:
            # 3. Вычисление гомографии методом RANSAC для нахождения глобального преобразования
            H, mask = cv2.findHomography(
                points, next_points,  # Исходные и целевые точки
                cv2.RANSAC,  # Метод RANSAC для устойчивой оценки
                ransacReprojThreshold=3.0,  # Порог ошибки репроекции (в пикселях)
                maxIters=2000,  # Максимальное количество итераций RANSAC
                confidence=0.995  # Доверительная вероятность
            )
        except cv2.error as e:
            print(f"Ошибка поиска гомографии: {e}")
            return False, np.zeros_like(flow)
        
        # Проверка успешности вычисления гомографии
        if H is None or mask is None:
            return False, np.zeros_like(flow)
        
        # Подсчет точек-выбросов (inliers), соответствующих модели
        inliers = np.sum(mask)
        # Проверка достаточности выбросов для валидации модели
        if inliers < self.min_inliers:
            return False, np.zeros_like(flow)
        
        # Создание координатной сетки для всего кадра
        try:
            coords = np.array([[x, y] for y in range(h) for x in range(w)], dtype=np.float32)
            
            # Применение гомографии ко всем точкам кадра
            warped_coords = cv2.perspectiveTransform(
                coords.reshape(-1, 1, 2), H
            ).reshape(-1, 2)
            
            # Вычисление ожидаемого потока для всего кадра
            expected_flow = warped_coords - coords
            # Преобразование в исходную форму (H, W, 2)
            expected_flow = expected_flow.reshape(h, w, 2)
            
            return True, expected_flow
        except Exception as e:
            print(f"Ошибка при вычислении ожидаемого потока: {e}")
            return False, np.zeros_like(flow)


def run_display_estimator():
    """
    Основная функция запуска и отображения карты глубины в реальном времени.
    
    Инициализирует оценщик глубины, захватывает видео с камеры,
    вычисляет и отображает карту глубины рядом с оригинальным кадром.
    """
    # Инициализация оценщика глубины
    estimator = CleanDisplayDepthEstimator()
    
    # Вывод информационного сообщения
    print("\n=== Построение карты глубины по движению одной камеры ===")
    print("      Перемещайте камеру для получения карты глубины.")
    print("      Нажмите 'q' для выхода\n")
    
    # Основной цикл обработки видео
    while True:
        # Захват кадра с камеры
        ret, frame = estimator.cap.read()
        if not ret:
            print("Ошибка чтения кадра с камеры")
            break
        
        # Изменение размера кадра для стабильности
        frame = cv2.resize(frame, (640, 480))
        
        # Получение карты глубины для текущего кадра
        depth_map = estimator.get_depth_map(frame)
        
        if depth_map is not None:
            # НОРМАЛИЗАЦИЯ ТОЛЬКО ТЕКУЩЕЙ КАРТЫ (без истории)
            # Использование процентилей для обрезки выбросов
            min_val = np.percentile(depth_map, 3)   # 3-й процентиль (отсекаем слишком темные)
            max_val = np.percentile(depth_map, 97)  # 97-й процентиль (отсекаем слишком яркие)
            
            # Нормализация к диапазону [0, 1] с обрезкой выбросов
            if max_val > min_val:
                depth_vis = np.clip((depth_map - min_val) / (max_val - min_val), 0, 1)
            else:
                depth_vis = np.zeros_like(depth_map)
            
            # Конвертация в 8-битное изображение (0-255)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            
            # Инвертирование цветов (ближние объекты становятся ярче)
            depth_map_inverted = 255 - depth_vis
            
            # Применение цветовой карты TURBO для наглядной визуализации
            depth_colored = cv2.applyColorMap(depth_map_inverted, cv2.COLORMAP_TURBO)
            
            # Индикация состояния системы
            status = "ACTIVE" if estimator.camera_moving else "STABLE"
            color = (0, 255, 0) if estimator.camera_moving else (255, 255, 0)
            
            # Добавление текста на карту глубины
            cv2.putText(depth_colored, f"Depth Map: {status}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Добавление текста на оригинальный кадр
            cv2.putText(frame, f"Camera: {status}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Горизонтальное объединение оригинального кадра и карты глубины
            combined = np.hstack((frame, depth_colored))
            cv2.imshow('Original | Clean Depth Display', combined)
        else:
            # Показываем только оригинальное видео при отсутствии данных о глубине
            cv2.putText(frame, "MOVE CAMERA", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Original | Clean Depth Display', frame)
        
        # Обработка нажатия клавиши 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождение ресурсов камеры и закрытие окон
    estimator.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    Точка входа в программу.
    """
    run_display_estimator()