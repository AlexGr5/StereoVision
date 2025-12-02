"""
Программа построения карты глубины с использованием стереокамер.
Основная идея: использует две камеры для захвата стереопар, применяет калибровку
и ректификацию, затем вычисляет карту глубины на основе параллакса (различий
между изображениями левой и правой камер).
"""

import cv2
import numpy as np
import os
import pickle
import time



#==============================================================================
# **************************
# КОНФИГУРАЦИОННЫЕ ПАРАМЕТРЫ
# **************************

# Файл с данными стереокалибровки (матрицы камер, коэффициенты искажений и т.д.)
STEREO_CALIBRATION_FILE = 'output/stereo_calibration_data.pkl'

# Разрешения, которые программа будет пытаться установить на камерах (в порядке приоритета)
DESIRED_RESOLUTIONS = [(1920, 1080), (1280, 720), (640, 480)]

# Размер окна для отображения изображений на экране
DISPLAY_SIZE = (640, 480)

# Параметры алгоритма стереосопоставления (StereoSGBM)
MIN_DISP = 16 * 0   # Минимальное смещение для поиска (может быть отрицательным)
NUM_DISP = 16 * 20  # Количество уровней диспаритета (должно быть кратно 16)
WINDOW_SIZE = 7     # Размер окна для сравнения блоков (нечетное число)
#==============================================================================




def ensure_same_size(left_img, right_img):
    """
    Приводит два изображения к одинаковому размеру (наименьшему из двух).
    
    Параметры:
    -----------
    left_img : numpy.ndarray
        Изображение с левой камеры
    right_img : numpy.ndarray
        Изображение с правой камеры
        
    Возвращает:
    -----------
    left_resized, right_resized : tuple
        Кортеж из двух изображений одинакового размера
    """
    h1, w1 = left_img.shape[:2]
    h2, w2 = right_img.shape[:2]
    
    # Если размеры уже одинаковые, возвращаем как есть
    if (h1, w1) == (h2, w2):
        return left_img, right_img
    
    # Используем минимальные размеры из двух изображений
    h_min = min(h1, h2)
    w_min = min(w1, w2)
    
    # Изменяем размеры обоих изображений
    left_resized = cv2.resize(left_img, (w_min, h_min))
    right_resized = cv2.resize(right_img, (w_min, h_min))
    
    print(f"Выровнены размеры: {w1}x{h1} и {w2}x{h2} -> {w_min}x{h_min}")
    return left_resized, right_resized


def find_available_cameras(max_test=10):
    """
    Находит все доступные камеры в системе.
    
    Параметры:
    -----------
    max_test : int
        Максимальный номер камеры для проверки
        
    Возвращает:
    -----------
    available_cameras : list
        Список словарей с информацией о каждой камере
    """
    available_cameras = []
    
    print("\n" + "="*60)
    print("ПОИСК И ПРОВЕРКА КАМЕР")
    print("="*60)
    print("\nИдет поиск камер...")
    print("Для каждой найденной камеры откроется окно предпросмотра.")
    print("Нажмите ESC для выхода из режима поиска.")
    
    # Проверяем каждую камеру
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Пробуем прочитать кадр
            ret, frame = cap.read()
            if ret:
                # Получаем параметры камеры
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"  Найдена камера {i}: {width}x{height} @ {fps:.1f} FPS")
                
                camera_info = {
                    'id': i,
                    'cap': cap,  # Сохраняем открытый объект захвата
                    'resolution': (width, height),
                    'fps': fps
                }
                
                available_cameras.append(camera_info)
            else:
                cap.release()
        else:
            continue
    
    print(f"\nНайдено камер: {len(available_cameras)}")
    
    # Если камеры найдены, показываем их в режиме предпросмотра
    if available_cameras:
        print("\nДля продолжения закройте все окна предпросмотра (ESC).")
        show_camera_previews(available_cameras)
    
    return available_cameras


def show_camera_previews(cameras, preview_time=10):
    """
    Показывает предпросмотр всех найденных камер.
    
    Параметры:
    -----------
    cameras : list
        Список доступных камер
    preview_time : int
        Максимальное время показа в секундах
    """
    start_time = time.time()
    
    while True:
        all_windows_closed = True
        frames_available = []
        
        # Читаем кадры со всех камер
        for cam in cameras:
            cap = cam['cap']
            ret, frame = cap.read()
            
            if ret:
                # Подготавливаем кадр для отображения
                display_frame = cv2.resize(frame, (320, 240))
                
                # Добавляем информацию
                cv2.putText(display_frame, f"Камера {cam['id']}", (10, 30), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "ESC - выход", (10, 220), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
                
                # Показываем кадр
                cv2.imshow(f'Camera {cam["id"]}', display_frame)
                all_windows_closed = False
                frames_available.append(cam['id'])
            else:
                # Если не удалось получить кадр, закрываем окно
                cv2.destroyWindow(f'Camera {cam["id"]}')
        
        # Проверяем, не истекло ли время
        if time.time() - start_time > preview_time:
            print(f"\nВремя предпросмотра истекло ({preview_time} секунд).")
            break
        
        # Проверяем нажатие ESC
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            print("\nРежим предпросмотра прерван пользователем.")
            break
        
        # Если все окна закрыты, выходим
        if all_windows_closed:
            break
        
        # Небольшая задержка для снижения нагрузки на CPU
        time.sleep(0.03)
    
    # Закрываем все окна
    for cam in cameras:
        cv2.destroyWindow(f'Camera {cam["id"]}')
    
    # Не освобождаем захват камеры, так как он еще понадобится
    print("Режим предпросмотра завершен.")


def select_cameras_visual(available_cameras):
    """
    Визуальный выбор двух камер из доступных.
    
    Параметры:
    -----------
    available_cameras : list
        Список доступных камер
        
    Возвращает:
    -----------
    (left_id, right_id) : tuple
        Кортеж с индексами выбранных камер (левая, правая)
    """
    if len(available_cameras) < 2:
        print(f"Ошибка: найдено только {len(available_cameras)} камер. Нужно минимум 2.")
        return None, None
    
    print("\n" + "="*60)
    print("ВИЗУАЛЬНЫЙ ВЫБОР КАМЕР")
    print("="*60)
    print("\nИнструкции:")
    print("1. Нажмите цифру от 0 до 9, чтобы выбрать камеру с соответствующим ID")
    print("2. Выберите две камеры для стереопары")
    print("3. ESC - отмена выбора")
    print("\nСписок доступных камер:")
    for cam in available_cameras:
        print(f"  Камера ID {cam['id']}")
    
    selected_ids = []
    
    # Создаем окно с инструкциями
    instructions = np.zeros((200, 600, 3), dtype=np.uint8)
    cv2.putText(instructions, "ВИЗУАЛЬНЫЙ ВЫБОР КАМЕР", (50, 40), 
               cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(instructions, "Нажмите цифру, чтобы выбрать камеру с таким ID", (50, 80), 
               cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(instructions, "ESC - отмена | ENTER - подтвердить выбор", (50, 110), 
               cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    
    while True:
        # Обновляем окно инструкций
        instructions_copy = instructions.copy()
        
        # Отображаем выбранные камеры
        if len(selected_ids) == 0:
            status = "Выбранные камеры: нет"
        elif len(selected_ids) == 1:
            status = f"Выбранные камеры: [{selected_ids[0]}]"
        else:
            status = f"Выбранные камеры: [{selected_ids[0]}, {selected_ids[1]}]"
        
        cv2.putText(instructions_copy, status, (50, 140), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
        
        # Подсказка
        if len(selected_ids) < 2:
            cv2.putText(instructions_copy, f"Выберите еще {2 - len(selected_ids)} камеру(ы)", (50, 170), 
                       cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.putText(instructions_copy, "Нажмите ENTER для подтверждения", (50, 170), 
                      cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Choise camers", instructions_copy)
        
        # Показываем предпросмотр доступных камер
        show_available_cameras_preview(available_cameras, selected_ids)
        
        # Обрабатываем клавиши
        key = cv2.waitKey(100) & 0xFF
        
        if key == 27:  # ESC
            print("Выбор отменен")
            break
        
        elif key == 13:  # ENTER
            if len(selected_ids) == 2:
                print(f"Выбраны камеры: {selected_ids}")
                cv2.destroyAllWindows()
                return selected_ids[0], selected_ids[1]
            else:
                print(f"Нужно выбрать 2 камеры. Сейчас выбрано: {len(selected_ids)}")
        
        # Проверяем цифры 0-9
        elif 48 <= key <= 57:  # Клавиши 0-9
            camera_id = key - 48  # Преобразуем ASCII в число
            
            # Проверяем, доступна ли камера с таким ID
            available_ids = [cam['id'] for cam in available_cameras]
            if camera_id in available_ids:
                if camera_id in selected_ids:
                    # Если камера уже выбрана, отменяем выбор
                    selected_ids.remove(camera_id)
                    print(f"Камера {camera_id} отменена")
                else:
                    if len(selected_ids) < 2:
                        selected_ids.append(camera_id)
                        print(f"Камера {camera_id} выбрана")
                    else:
                        print("Уже выбрано 2 камеры. Сначала отмените одну.")
            else:
                print(f"Камера {camera_id} не доступна")
        
        # Проверяем закрытие окна
        if cv2.getWindowProperty("Choise camers", cv2.WND_PROP_VISIBLE) < 0:
            print("Окно выбора закрыто")
            break
    
    cv2.destroyAllWindows()
    return None, None


def show_available_cameras_preview(cameras, selected_ids):
    """
    Показывает предпросмотр доступных камер с выделением выбранных.
    
    Параметры:
    -----------
    cameras : list
        Список доступных камер
    selected_ids : list
        Список ID выбранных камер
    """
    for cam in cameras:
        cap = cam['cap']
        camera_id = cam['id']
        
        # Пробуем прочитать кадр
        ret, frame = cap.read()
        if not ret:
            # Если не удалось прочитать кадр, пропускаем эту камеру
            continue
        
        # Подготавливаем кадр для отображения
        display_frame = cv2.resize(frame, (320, 240))
        
        # Если камера выбрана, добавляем рамку
        if camera_id in selected_ids:
            idx = selected_ids.index(camera_id)
            color = (0, 255, 0) if idx == 0 else (255, 0, 0)  # Зеленый для первой, красный для второй
            cv2.rectangle(display_frame, (0, 0), (319, 239), color, 3)
            cv2.putText(display_frame, f"Выбрана #{idx+1}", (10, 220), 
                       cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)
        
        # Добавляем информацию о камере
        cv2.putText(display_frame, f"ID: {camera_id}", (10, 30), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Нажмите {camera_id}", (10, 60), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
        
        # Показываем кадр
        window_name = f"Preview camera {camera_id}"
        cv2.imshow(window_name, display_frame)


def determine_left_right_visual(cam1_id, cam2_id):
    """
    Визуальный интерфейс для определения левой и правой камер.
    
    Параметры:
    -----------
    cam1_id, cam2_id : int
        Индексы двух выбранных камер
        
    Возвращает:
    -----------
    (left_id, right_id) : tuple
        Кортеж с индексами левой и правой камер
    """
    print(f"\nОпределение расположения камер {cam1_id} и {cam2_id}...")
    
    # Открываем камеры для определения
    cap1 = cv2.VideoCapture(cam1_id)
    cap2 = cv2.VideoCapture(cam2_id)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Ошибка: не удалось открыть одну из камер")
        return None, None
    
    # Устанавливаем параметры
    for cap in [cap1, cap2]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    selected_left = None
    selected_right = None
    
    print("\nИнструкции:")
    print("  'L' - если слева ЛЕВАЯ камера")
    print("  'R' - если слева ПРАВАЯ камера")
    print("  'Q' - отмена выбора")
    print("\nРазместите объект перед камерами и определите правильное расположение.")
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            continue
        
        # Создаем комбинированное изображение
        combined = np.hstack((frame1, frame2))
        
        # Добавляем подписи
        cv2.putText(combined, f"Камера {cam1_id}", (10, 30), 
                   cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, f"Камера {cam2_id}", (650, 30), 
                   cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        
        # Добавляем инструкции
        cv2.putText(combined, "Определите ЛЕВУЮ и ПРАВУЮ камеры:", (10, 450), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(combined, "L - слева ЛЕВАЯ | R - слева ПРАВАЯ | Q - отмена", (10, 480), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)
        
        cv2.imshow('Detect left/right camera', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('l'):
            selected_left, selected_right = cam1_id, cam2_id
            print(f"Выбрано: Левая камера = {cam1_id}, Правая камера = {cam2_id}")
            break
        elif key == ord('r'):
            selected_left, selected_right = cam2_id, cam1_id
            print(f"Выбрано: Левая камера = {cam2_id}, Правая камера = {cam1_id}")
            break
        elif key == ord('q'):
            print("Определение отменено")
            break
        
        # Проверяем закрытие окна
        if cv2.getWindowProperty('Detect left/right camera', cv2.WND_PROP_VISIBLE) < 0:
            print("Окно закрыто. Отмена.")
            break
    
    # Освобождаем ресурсы
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    
    return selected_left, selected_right


def select_cameras_interactive_visual():
    """
    Полный интерактивный процесс выбора камер с визуальным интерфейсом.
    
    Возвращает:
    -----------
    (left_id, right_id) : tuple
        Кортеж с индексами левой и правой камер
    """
    print("\n" + "="*70)
    print("ВИЗУАЛЬНЫЙ ВЫБОР СТЕРЕОКАМЕР")
    print("="*70)
    
    # Шаг 1: Поиск камер
    print("\nШАГ 1: Поиск доступных камер...")
    available_cameras = find_available_cameras()
    
    if len(available_cameras) < 2:
        print(f"\nНедостаточно камер! Найдено: {len(available_cameras)}")
        print("Проверьте подключение камер и повторите попытку.")
        
        # Освобождаем ресурсы камер
        for cam in available_cameras:
            cam['cap'].release()
        
        return None, None
    
    # Шаг 2: Визуальный выбор двух камер
    print("\nШАГ 2: Визуальный выбор двух камер...")
    cam1_id, cam2_id = select_cameras_visual(available_cameras)
    
    # Освобождаем ресурсы камер, которые не были выбраны
    for cam in available_cameras:
        cam_id = cam['id']
        if cam_id != cam1_id and cam_id != cam2_id:
            cam['cap'].release()
    
    if cam1_id is None or cam2_id is None:
        print("Выбор камер отменен.")
        # Освобождаем оставшиеся ресурсы
        for cam in available_cameras:
            if cam['id'] == cam1_id or cam['id'] == cam2_id:
                cam['cap'].release()
        return None, None
    
    # Шаг 3: Определение левой/правой камеры
    print("\nШАГ 3: Определение левой и правой камеры...")
    left_id, right_id = determine_left_right_visual(cam1_id, cam2_id)
    
    # Освобождаем оставшиеся ресурсы
    for cam in available_cameras:
        if cam['id'] == cam1_id or cam['id'] == cam2_id:
            cam['cap'].release()
    
    if left_id is None or right_id is None:
        print("Определение расположения отменено.")
        return None, None
    
    # Шаг 4: Подтверждение выбора
    print("\n" + "-"*70)
    print("ВЫБОР ЗАВЕРШЕН")
    print(f"  Левая камера: ID {left_id}")
    print(f"  Правая камера: ID {right_id}")
    print("-"*70)
    
    """
    # Предлагаем протестировать выбранную пару
    print("\nХотите протестировать выбранную стереопару?")
    print("1. Да, протестировать (5 секунд)")
    print("2. Нет, сразу начать работу")
    
    choice = input("Ваш выбор (1/2): ").strip()
    
    if choice == '1':
        print("\nТестирование стереопары...")
        test_stereo_pair_visual(left_id, right_id, test_time=5)
    """
    
    
    return left_id, right_id


def test_stereo_pair_visual(left_id, right_id, test_time=5):
    """
    Тестирование выбранной стереопары с визуальным интерфейсом.
    
    Параметры:
    -----------
    left_id, right_id : int
        Индексы левой и правой камер
    test_time : int
        Время тестирования в секундах
    """
    left_cap = cv2.VideoCapture(left_id)
    right_cap = cv2.VideoCapture(right_id)
    
    if not left_cap.isOpened() or not right_cap.isOpened():
        print("Ошибка: не удалось открыть одну из камер")
        return
    
    # Устанавливаем параметры
    for cap in [left_cap, right_cap]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\nТЕСТИРОВАНИЕ СТЕРЕОПАРЫ")
    print("Разместите объект перед камерами для проверки.")
    print("Нажмите 'Q' для досрочного завершения теста.")
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < test_time:
        ret_left, frame_left = left_cap.read()
        ret_right, frame_right = right_cap.read()
        
        if not ret_left or not ret_right:
            print("Ошибка получения кадра")
            break
        
        frame_count += 1
        
        # Создаем комбинированное изображение
        combined = np.hstack((frame_left, frame_right))
        
        # Добавляем информацию
        cv2.putText(combined, f"Левая камера (ID: {left_id})", (10, 30), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, f"Правая камера (ID: {right_id})", (650, 30), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, f"Кадр: {frame_count}", (10, 450), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)
        
        cv2.imshow('Тестирование стереопары', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Тест прерван пользователем")
            break
    
    left_cap.release()
    right_cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nТестирование завершено.")
    print(f"Обработано кадров: {frame_count}")
    print(f"Средняя частота кадров: {frame_count/(test_time if frame_count > 0 else 1):.1f} FPS")

def load_stereo_calibration():
    """
    Загружает данные стереокалибровки из файла и готовит карты ректификации.
    
    Возвращает:
    -----------
    calibration_result : dict or None
        Словарь с данными калибровки и картами ректификации,
        или None если загрузка не удалась
    """
    if not os.path.exists(STEREO_CALIBRATION_FILE):
        print(f"❌ Ошибка: файл стереокалибровки не найден: {STEREO_CALIBRATION_FILE}")
        return None
    
    try:
        # Загружаем данные калибровки из pickle файла
        with open(STEREO_CALIBRATION_FILE, 'rb') as f:
            stereo_data = pickle.load(f)
        
        print("✅ Данные стереокалибровки успешно загружены")
        
        # Извлекаем параметры
        mtx_left = stereo_data['mtx_left']      # Матрица камеры левой камеры
        dist_left = stereo_data['dist_left']    # Коэффициенты искажения левой камеры
        mtx_right = stereo_data['mtx_right']    # Матрица камеры правой камеры
        dist_right = stereo_data['dist_right']  # Коэффициенты искажения правой камеры
        R = stereo_data['R']                    # Матрица вращения между камерами
        T = stereo_data['T']                    # Вектор трансляции между камерами
        img_size = tuple(stereo_data['img_size'])  # Размер изображения при калибровке
        
        print(f"Калибровочный размер: {img_size}")
        
        # Выполняем стереоректификацию
        # alpha=0 означает обрезку черных областей после ректификации
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx_left, dist_left,
            mtx_right, dist_right,
            img_size, R, T,
            alpha=0  # Автоматическая обрезка черных областей
        )
        
        # Создаем карты ректификации для быстрого преобразования
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, R1, P1, img_size, cv2.CV_16SC2
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            mtx_right, dist_right, R2, P2, img_size, cv2.CV_16SC2
        )
        
        # Формируем результат
        calibration_result = {
            'left_map1': left_map1,
            'left_map2': left_map2,
            'right_map1': right_map1,
            'right_map2': right_map2,
            'roi1': roi1,      # Region of Interest для левой камеры
            'roi2': roi2,      # Region of Interest для правой камеры
            'Q': Q,            # Матрица репроекции для 3D реконструкции
            'R': R,
            'T': T,
            'mtx_left': mtx_left,
            'dist_left': dist_left,
            'mtx_right': mtx_right,
            'dist_right': dist_right,
            'baseline': abs(T[0, 0]),  # Базовое расстояние между камерами (в метрах)
            'img_size': img_size
        }
        
        print(f"Стереоректификация подготовлена. Базис: {abs(T[0, 0]):.4f} м")
        print(f"ROI левой камеры: {roi1}")
        print(f"ROI правой камеры: {roi2}")
        return calibration_result
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке стереокалибровки: {e}")
        import traceback
        traceback.print_exc()
        return None


def set_camera_resolution(cap, desired_resolutions):
    """
    Устанавливает максимальное поддерживаемое разрешение на камере.
    
    Параметры:
    -----------
    cap : cv2.VideoCapture
        Объект захвата видео
    desired_resolutions : list
        Список разрешений в порядке предпочтения
        
    Возвращает:
    -----------
    (actual_width, actual_height), success : tuple
        Фактическое разрешение и флаг успеха
    """
    for width, height in desired_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if (actual_width, actual_height) == (width, height):
            print(f"Установлено разрешение: {width}x{height}")
            return (width, height), True
    
    # Если ни одно из желаемых разрешений не поддерживается
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Используется разрешение по умолчанию: {actual_width}x{actual_height}")
    return (actual_width, actual_height), False


def initialize_cameras(left_id, right_id, target_size=None):
    """
    Инициализирует обе камеры с одинаковыми параметрами.
    
    Параметры:
    -----------
    left_id : int
        Индекс левой камеры
    right_id : int
        Индекс правой камеры
    target_size : tuple, optional
        Желаемый размер изображения (ширина, высота)
        
    Возвращает:
    -----------
    left_cap, right_cap : tuple
        Объекты VideoCapture для обеих камер
    """
    left_cap = cv2.VideoCapture(left_id)
    right_cap = cv2.VideoCapture(right_id)
    
    if not left_cap.isOpened() or not right_cap.isOpened():
        print("Ошибка: Не удалось открыть одну из камер")
        return None, None
    
    # Устанавливаем одинаковые параметры для обеих камер
    for cap in [left_cap, right_cap]:
        cap.set(cv2.CAP_PROP_FPS, 30)           # Частота кадров
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)      # Отключаем автофокус
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Режим экспозиции
        cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)     # Значение экспозиции
    
    # Если указан целевой размер, пытаемся установить его
    if target_size:
        width, height = target_size
        left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Получаем фактическое разрешение
    left_width = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    left_height = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    right_width = int(right_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    right_height = int(right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Левая камера: {left_width}x{left_height}")
    print(f"Правая камера: {right_width}x{right_height}")
    
    return left_cap, right_cap


def resize_to_target(frame, target_size):
    """
    Изменяет размер изображения до целевого.
    
    Параметры:
    -----------
    frame : numpy.ndarray
        Входное изображение
    target_size : tuple
        Целевой размер (ширина, высота)
        
    Возвращает:
    -----------
    resized_frame : numpy.ndarray
        Изображение измененного размера
    """
    if frame.shape[1] == target_size[0] and frame.shape[0] == target_size[1]:
        return frame
    
    return cv2.resize(frame, target_size)


def apply_stereo_rectification(left_img, right_img, stereo_calib):
    """
    Применяет стереоректификацию к изображениям.
    
    Ректификация выравнивает изображения так, что соответственные точки
    находятся на одной горизонтальной линии (эпиполярной линии).
    
    Параметры:
    -----------
    left_img : numpy.ndarray
        Изображение с левой камеры
    right_img : numpy.ndarray
        Изображение с правой камеры
    stereo_calib : dict
        Данные стереокалибровки
        
    Возвращает:
    -----------
    left_rectified, right_rectified : tuple
        Ректифицированные изображения
    """
    if stereo_calib is None:
        return left_img, right_img
    
    try:
        # Получаем текущий и калибровочный размеры
        current_height, current_width = left_img.shape[:2]
        calib_width, calib_height = stereo_calib['img_size']
        
        # Если размеры не совпадают, изменяем размер изображений
        if (current_width, current_height) != (calib_width, calib_height):
            left_img = resize_to_target(left_img, (calib_width, calib_height))
            right_img = resize_to_target(right_img, (calib_width, calib_height))
            print(f"Изменен размер изображений до калибровочного: {calib_width}x{calib_height}")
        
        # Применяем ректификацию с использованием предвычисленных карт
        left_rectified = cv2.remap(
            left_img, 
            stereo_calib['left_map1'], 
            stereo_calib['left_map2'], 
            cv2.INTER_LINEAR
        )
        right_rectified = cv2.remap(
            right_img, 
            stereo_calib['right_map1'], 
            stereo_calib['right_map2'], 
            cv2.INTER_LINEAR
        )
        
        # Примечание: обрезка по ROI закомментирована, так как при alpha=0
        # в stereoRectify уже получены оптимальные области
        return left_rectified, right_rectified
        
    except Exception as e:
        print(f"Ошибка при стереоректификации: {e}")
        return left_img, right_img


def create_depth_map(left_img, right_img, stereo_calib=None, min_depth=0.3, max_depth=2.0):
    """
    Создает карту глубины из стереопары.
    
    Принцип работы: находит соответственные точки на двух изображениях,
    вычисляет смещение (диспаритет), затем преобразует в глубину по формуле:
    глубина = (f * B) / диспаритет, где f - фокусное расстояние, B - базис.
    
    Параметры:
    -----------
    left_img : numpy.ndarray
        Изображение с левой камеры
    right_img : numpy.ndarray
        Изображение с правой камеры
    stereo_calib : dict, optional
        Данные калибровки для точных вычислений
    min_depth : float
        Минимальная глубина для отображения (в метрах)
    max_depth : float
        Максимальная глубина для отображения (в метрах)
        
    Возвращает:
    -----------
    depth_final : numpy.ndarray
        Карта глубины в метрах
    disparity : numpy.ndarray
        Карта диспаритета (смещения в пикселях)
    depth_colormap : numpy.ndarray
        Цветное представление карты глубины для визуализации
    """
    # Дополнительная проверка размеров
    left_img, right_img = ensure_same_size(left_img, right_img)
    
    # Конвертируем в grayscale (алгоритм StereoSGBM работает с grayscale)
    if len(left_img.shape) == 3:
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_left = left_img
        
    if len(right_img.shape) == 3:
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_right = right_img
    
    # Убеждаемся, что изображения 8-битные
    if gray_left.dtype != np.uint8:
        gray_left = np.uint8(np.clip(gray_left, 0, 255))
    if gray_right.dtype != np.uint8:
        gray_right = np.uint8(np.clip(gray_right, 0, 255))
    
    # Проверка размеров
    if gray_left.shape != gray_right.shape:
        print(f"ОШИБКА: Размеры после конвертации не совпадают: {gray_left.shape} vs {gray_right.shape}")
        gray_left, gray_right = ensure_same_size(gray_left, gray_right)
    
    try:
        # Настройка алгоритма StereoSGBM для вычисления диспаритета
        stereo = cv2.StereoSGBM_create(
            minDisparity=MIN_DISP,      # Минимальное смещение
            numDisparities=NUM_DISP,    # Количество уровней диспаритета
            blockSize=WINDOW_SIZE,      # Размер окна сравнения
            P1=8 * 3 * WINDOW_SIZE**2,  # Параметр сглаживания 1
            P2=32 * 3 * WINDOW_SIZE**2, # Параметр сглаживания 2
            disp12MaxDiff=1,            # Максимальная разница в диспаритете
            uniquenessRatio=10,         # Порог уникальности соответствия
            speckleWindowSize=100,      # Размер окна для фильтрации шума
            speckleRange=32,            # Диапазон для фильтрации шума
            preFilterCap=63,            # Предел предварительной фильтрации
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Режим вычислений
        )
        
        # Вычисляем диспаритет (значения в целых пикселях, делим на 16 для получения действительных значений)
        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # Фильтруем шумы медианным фильтром
        disparity = cv2.medianBlur(disparity, 5)
        
        # Вычисляем глубину по формуле: глубина = (f * B) / диспаритет
        if stereo_calib is not None and 'calibration_data' in stereo_calib:
            # Используем калибровочные данные
            fx = stereo_calib['calibration_data']['camera_matrix'][0, 0]
        else:
            # Значение по умолчанию
            fx = 700
        
        # Базовое расстояние между камерами (в метрах)
        BASELINE = 0.08
        
        # Расчет глубины (добавлена защита от деления на ноль)
        depth = (fx * BASELINE) / (disparity + 1e-6)
        
        # Ограничиваем глубину диапазоном
        depth_clipped = np.clip(depth, min_depth, max_depth)
        
        # Создаем маску валидных значений (где диспаритет корректен и глубина в диапазоне)
        valid_mask = (disparity > MIN_DISP) & (depth_clipped >= min_depth) & (depth_clipped <= max_depth)
        depth_final = np.where(valid_mask, depth_clipped, 0)
        
        # Нормализация для визуализации
        depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
        
        return depth_final, disparity, depth_colormap
        
    except Exception as e:
        print(f"Ошибка при создании карты глубины: {e}")
        h, w = gray_left.shape
        empty_uint8 = np.zeros((h, w), dtype=np.uint8)
        empty_colormap = cv2.applyColorMap(empty_uint8, cv2.COLORMAP_TURBO)
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32), empty_colormap


def put_multiline_text_anywhere(img, text, position='bottom', font_face=cv2.FONT_HERSHEY_COMPLEX, 
                               font_scale=0.6, color=(255, 128, 0), thickness=2, 
                               margin=20, line_spacing=30):
    """
    Выводит многострочный текст на изображение с автоматическим переносом.
    
    Параметры:
    -----------
    img : numpy.ndarray
        Изображение для рисования текста
    text : str
        Текст для отображения
    position : str
        Позиция текста: 'top', 'bottom', 'center'
    font_face : int
        Шрифт OpenCV
    font_scale : float
        Масштаб шрифта
    color : tuple
        Цвет текста в формате BGR
    thickness : int
        Толщина линии текста
    margin : int
        Отступ от края изображения
    line_spacing : int
        Межстрочный интервал
        
    Возвращает:
    -----------
    lines_drawn : int
        Количество отрисованных строк
    """
    img_height, img_width = img.shape[:2]
    max_width = img_width - 2 * margin  # Максимальная ширина строки
    
    # Разбиваем текст на строки по словам
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        (text_width, _), _ = cv2.getTextSize(test_line, font_face, font_scale, thickness)
        
        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Определяем стартовую позицию Y в зависимости от положения
    total_height = len(lines) * line_spacing
    
    if position == 'top':
        start_y = margin + line_spacing
    elif position == 'bottom':
        start_y = img_height - margin - total_height
        # Если не помещается снизу, поднимаем выше
        if start_y < margin:
            start_y = margin
    elif position == 'center':
        start_y = (img_height - total_height) // 2
    else:
        start_y = margin  # по умолчанию сверху
    
    # Выводим строки на изображение
    for i, line in enumerate(lines):
        y = start_y + i * line_spacing
        # Проверяем, не выходит ли строка за пределы изображения
        if y > img_height - margin:
            break  # прекращаем вывод, если вышли за нижнюю границу
        cv2.putText(img, line, (margin, y), font_face, font_scale, color, thickness)
    
    return len(lines)


def main():
    """
    Главная функция программы.
    """
    print("=== ПРОГРАММА ПОСТРОЕНИЯ КАРТЫ ГЛУБИНЫ ===")
    
    # Инициализируем параметры глубины
    min_depth = 0.2
    max_depth = 4.0
    
    """
    # Выбор камер
    print("\nРЕЖИМЫ ВЫБОРА КАМЕР:")
    print("   [1] Визуальный выбор (рекомендуется)")
    print("   [2] Ручной ввод ID камер")
    print("   [3] Использовать сохраненные настройки")
    
    camera_choice = input("\nВыберите режим (1-3): ").strip()
    """
    #!!!!!!
    # ВСЕГДА ВИЗУАЛЬНЫЙ ВЫБОР
    camera_choice = '1'
    
    if camera_choice == '1':
        # Визуальный выбор камер
        LEFT_CAMERA_ID, RIGHT_CAMERA_ID = select_cameras_interactive_visual()
        
        if LEFT_CAMERA_ID is None or RIGHT_CAMERA_ID is None:
            print("Не удалось выбрать камеры. Выход.")
            return
            
    elif camera_choice == '2':
        # Ручной ввод
        print("\nРУЧНОЙ ВВОД ID КАМЕР")
        print("Сначала найдем доступные камеры...")
        
        available_cameras = find_available_cameras()
        available_ids = [cam['id'] for cam in available_cameras]
        
        # Освобождаем ресурсы камер
        for cam in available_cameras:
            cam['cap'].release()
        
        if len(available_ids) < 2:
            print("Недостаточно камер для ручного ввода!")
            return
        
        print(f"\nДоступные камеры: {available_ids}")
        
        try:
            left_id = int(input("Введите ID левой камеры: "))
            right_id = int(input("Введите ID правой камеры: "))
            
            if left_id not in available_ids or right_id not in available_ids:
                print("Ошибка: одна или обе камеры недоступны!")
                return
            
            # Определяем левую/правую
            LEFT_CAMERA_ID, RIGHT_CAMERA_ID = determine_left_right_visual(left_id, right_id)
            
        except ValueError:
            print("Ошибка: введите целые числа!")
            return
            
    elif camera_choice == '3':
        # Использование сохраненных настроек
        print("Режим использования сохраненных настроек пока не реализован.")
        print("Используем визуальный выбор.")
        LEFT_CAMERA_ID, RIGHT_CAMERA_ID = select_cameras_interactive_visual()
    else:
        print("Неверный выбор. Используем визуальный выбор.")
        LEFT_CAMERA_ID, RIGHT_CAMERA_ID = select_cameras_interactive_visual()
    
    if LEFT_CAMERA_ID is None or RIGHT_CAMERA_ID is None:
        print("Не удалось определить камеры. Выход.")
        return
    
    print(f"\nИнициализация камер: Левая={LEFT_CAMERA_ID}, Правая={RIGHT_CAMERA_ID}")
    
    # Загружаем стереокалибровку
    stereo_calib = load_stereo_calibration()
    
    # Определяем целевой размер для камер
    target_size = stereo_calib['img_size'] if stereo_calib else (640, 480)
    print("Целевой размер для камер: ", target_size)
    
    # Инициализируем камеры
    left_cap, right_cap = initialize_cameras(LEFT_CAMERA_ID, RIGHT_CAMERA_ID, target_size)
    
    if left_cap is None or right_cap is None:
        print("Ошибка инициализации камер. Выход.")
        return

    # Настройки отображения
    use_stereo_rectify = stereo_calib is not None  # Использовать ректификацию если есть калибровка
    show_depth = False    # Показывать ли карту глубины
    show_disparity = False  # Показывать ли карту диспаритета

    print("\n=== УПРАВЛЕНИЕ ===")
    print("'q' - выход")
    print("'s' - сохранить текущий кадр")
    print("'z' - переключить карту глубины")
    print("'x' - переключить карту диспаритета")
    if use_stereo_rectify:
        print("'r' - переключить стереоректификацию")
    print("'1'/'2' - регулировать минимальную глубину")
    print("'3'/'4' - регулировать максимальную глубину")

    try:
        while True:
            # Захват кадров с обеих камер
            ret_left, frame_left = left_cap.read()
            ret_right, frame_right = right_cap.read()
        
            if not ret_left or not ret_right:
                print("Ошибка: Не удалось получить кадр с камеры")
                break
            
            # Обработка изображений
            processed_left = frame_left.copy()
            processed_right = frame_right.copy()
            
            # Применение стереоректификации
            if use_stereo_rectify:
                processed_left, processed_right = apply_stereo_rectification(
                    processed_left, processed_right, stereo_calib
                )
            
            # Подготовка изображений для отображения
            display_left = cv2.resize(processed_left, DISPLAY_SIZE)
            display_right = cv2.resize(processed_right, DISPLAY_SIZE)
            
            # Добавляем информацию о статусе
            status_text = "Стереорект." if use_stereo_rectify else "Сырые кадры"
            
            cv2.putText(display_left, f"Левая | {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_right, f"Правая | {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
            
            # Отображаем стереопару
            combined = np.hstack((display_left, display_right))
            
            # Выводим подсказки управления
            instruction_text = "'q' - выход | 's' - сохранить текущий кадр | 'z' - переключить карту глубины | 'x' - переключить карту диспаритета | 'r' - переключить стереоректификацию | '1'/'2' - регулировать минимальную глубину | '3'/'4' - регулировать максимальную глубину"
            put_multiline_text_anywhere(combined, instruction_text, position='bottom')
            
            cv2.imshow('Stereo cameras', combined)
            
            # Обработка карты глубины/диспаритета (если включено)
            if show_depth or show_disparity:
                # Получаем карту глубины
                depth, disparity, depth_colormap = create_depth_map(
                    processed_left, processed_right, stereo_calib, min_depth, max_depth
                )
                
                if show_depth:
                    # Нормализуем глубину для отображения
                    depth_display = np.zeros_like(depth)
                    valid_mask = depth > 0
                    if np.any(valid_mask):
                        depth_display[valid_mask] = np.clip(depth[valid_mask], min_depth, max_depth)
                        depth_display = ((depth_display - min_depth) / 
                                      (max_depth - min_depth) * 255).astype(np.uint8)
                    
                    # Применение цветовой карты
                    depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_TURBO)
                    
                    # Добавляем информацию о диапазоне глубины
                    depth_info = f"Глубина: {min_depth:.1f}-{max_depth:.1f}м"
                    cv2.putText(depth_colormap, depth_info, (10, 30),
                               cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                    
                    depth_display_resized = cv2.resize(depth_colormap, DISPLAY_SIZE)
                    cv2.imshow('Depth map', depth_display_resized)
                
                if show_disparity:
                    # Нормализуем диспаритет для отображения
                    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
                    disparity_normalized = disparity_normalized.astype(np.uint8)
                    disparity_colormap = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
                    
                    # Добавляем информацию о диспаритете
                    disp_info = f"Диспаритет: {NUM_DISP}px"
                    cv2.putText(disparity_colormap, disp_info, (10, 30),
                               cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                    
                    disparity_display = cv2.resize(disparity_colormap, DISPLAY_SIZE)
                    cv2.imshow('Disparity map', disparity_display)
            
            # Обработка клавиш управления
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and stereo_calib is not None:
                use_stereo_rectify = not use_stereo_rectify
                print(f"Стереоректификация: {'ВКЛ' if use_stereo_rectify else 'ВЫКЛ'}")
            elif key == ord('s'):
                # Сохранение текущих кадров
                timestamp = int(time.time() * 1000)
                os.makedirs("depth_captures", exist_ok=True)
                cv2.imwrite(f"depth_captures/left_{timestamp}.jpg", processed_left)
                cv2.imwrite(f"depth_captures/right_{timestamp}.jpg", processed_right)
                if show_depth or show_disparity:
                    cv2.imwrite(f"depth_captures/depth_{timestamp}.jpg", depth_colormap if show_depth else disparity_colormap)
                print(f"Кадры сохранены с timestamp: {timestamp}")
            elif key == ord('z'):
                show_disparity = not show_disparity
                if show_disparity:
                    show_depth = False
                print(f"Карта глубины: {'ВКЛ' if show_depth else 'ВЫКЛ'}")
            elif key == ord('x'):
                show_depth = not show_depth
                if show_depth:
                    show_disparity = False
                print(f"Карта диспаритета: {'ВКЛ' if show_disparity else 'ВЫКЛ'}")
            elif key == ord('1'):
                min_depth = max(0.1, min_depth - 0.1)
                print(f"Минимальная глубина: {min_depth:.1f}м")
            elif key == ord('2'):
                min_depth += 0.1
                print(f"Минимальная глубина: {min_depth:.1f}м")
            elif key == ord('3'):
                max_depth = max(min_depth + 0.1, max_depth - 0.1)
                print(f"Максимальная глубина: {max_depth:.1f}м")
            elif key == ord('4'):
                max_depth += 0.1
                print(f"Максимальная глубина: {max_depth:.1f}м")
        
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Освобождаем ресурсы
        left_cap.release()
        right_cap.release()
        cv2.destroyAllWindows()
        print("Ресурсы освобождены")
        print("⚠️НИЖЕ ОТОБРАЗЯТСЯ ЛОГИ OpenCV. ЭТО НЕ ОШИБКИ!⚠️")
        


if __name__ == "__main__":
    main()