import pygame
import pygame.camera
import sys
import time
import os
import numpy as np
import cv2
import pickle
import glob
import threading
import queue
import time

# Импортируем программы для калибровок
from camera_calibration import invoke   # Калибровка по отдельности
import stereo_calibration               # Калибровка стереопары после отдельных калибровко
import universal_stereo_calibration     # Универсальная калибровка

#===========================================================================================================
# ОБЯЗАТЕЛЬНО УКАЗАТЬ ВЕРНЫЕ ДАННЫЕ!
# Константы для калибровки:
CHESSBOARD_SIZE = (9, 6)    # Количество пересечений шахматной доски (7, 7) (9, 6)
SQUARE_SIZE = 5.86          # Размер квадрата в санитиметрах (2.65) (8.35)
#===========================================================================================================



class CalibrationThread(threading.Thread):
    """
    Класс потока для выполнения калибровки в фоновом режиме.
    
    Позволяет выполнять длительные операции калибровки без блокировки
    основного интерфейса пользователя.
    """
    
    def __init__(self, calibration_type, params=None):
        """
        Инициализация потока калибровки.
        
        Параметры:
        -----------
        calibration_type : str
            Тип калибровки: 'individual', 'stereo' или 'universal'
        params : dict, optional
            Дополнительные параметры калибровки (размер шахматной доски и т.д.)
            
        Атрибуты:
        -----------
        result : dict или None
            Результаты калибровки или None, если калибровка не завершена
        error : str или None
            Сообщение об ошибке, если калибровка не удалась
        progress : int
            Процент выполнения калибровки (0-100)
        status_message : str
            Текущее статусное сообщение для отображения
        is_running : bool
            Флаг активности потока
        """
        # Вызов конструктора родительского класса threading.Thread
        threading.Thread.__init__(self)
        
        # Тип калибровки: индивидуальная, стерео или универсальная
        self.calibration_type = calibration_type
        
        # Параметры калибровки (используются значения по умолчанию, если не заданы)
        self.params = params or {}
        
        # Результат калибровки (заполняется после завершения)
        self.result = None
        
        # Сообщение об ошибке (заполняется при возникновении исключения)
        self.error = None
        
        # Прогресс выполнения в процентах (0-100)
        self.progress = 0
        
        # Текущее статусное сообщение для отображения пользователю
        self.status_message = ""
        
        # Флаг активности потока (используется для остановки)
        self.is_running = True
        
        # Установка потока как демона (завершается при завершении главного потока)
        self.daemon = True
        
    def run(self):
        """
        Основной метод потока, вызывается при запуске start().
        
        Выполняет калибровку в зависимости от выбранного типа.
        Обрабатывает исключения и сохраняет ошибки для последующего анализа.
        """
        try:
            # Выбор метода калибровки в зависимости от типа
            if self.calibration_type == 'individual':
                self.run_individual_calibration()
            elif self.calibration_type == 'stereo':
                self.run_stereo_calibration()
            elif self.calibration_type == 'universal':
                self.run_universal_calibration()
        except Exception as e:
            # Сохранение сообщения об ошибке
            self.error = str(e)
            # Вывод трассировки стека для отладки
            import traceback
            traceback.print_exc()
    
    def run_individual_calibration(self):
        """
        Индивидуальная калибровка камер (левой и правой отдельно).
        
        Выполняет калибровку каждой камеры по отдельности с использованием
        изображений шахматной доски, затем проверяет качество калибровки.
        """
        # Первый этап: калибровка левой камеры
        self.status_message = "Калибровка левой камеры..."
        self.progress = 25  # Установка прогресса на 25%
        
        # Вызов функции калибровки для левой камеры
        # 'captures' - папка с изображениями, 'left_*.jpg' - шаблон имен файлов
        invoke('captures', 'left_*.jpg', 'output', 'left', 
               self.params.get('chessboard_size', CHESSBOARD_SIZE),  # Размер шахматной доски (по умолчанию из константы)
               self.params.get('square_size', SQUARE_SIZE))  # Размер квадрата доски в мм
        
        # Второй этап: калибровка правой камеры
        self.status_message = "Калибровка правой камеры..."
        self.progress = 50  # Установка прогресса на 50%
        
        # Вызов функции калибровки для правой камеры
        invoke('captures', 'right_*.jpg', 'output', 'right', 
               self.params.get('chessboard_size', CHESSBOARD_SIZE), 
               self.params.get('square_size', SQUARE_SIZE))
        
        # Третий этап: проверка качества калибровки
        self.status_message = "Проверка качества калибровки..."
        self.progress = 75  # Установка прогресса на 75%
        
        # Проверка качества калибровки по метрике RMS (Root Mean Square)
        left_rms = check_calibration_quality('output/calibration_data_left.pkl')
        right_rms = check_calibration_quality('output/calibration_data_right.pkl')
        
        # Завершающий этап
        self.status_message = "Завершение..."
        self.progress = 100  # Установка прогресса на 100%
        
        # Формирование результата калибровки
        # RMS < 3.0 считается хорошим результатом
        self.result = {
            'left_rms': left_rms,   # Среднеквадратичная ошибка для левой камеры
            'right_rms': right_rms, # Среднеквадратичная ошибка для правой камеры
            'success': left_rms < 3.0 and right_rms < 3.0  # Флаг успешности калибровки
        }
    
    def run_stereo_calibration(self):
        """
        Стереокалибровка для пары камер.
        
        Выполняет совместную калибровку двух камер для определения
        их взаимного расположения и параметров стереопары.
        """
        self.status_message = "Проверка изображений..."
        self.progress = 20  # Установка прогресса на 20%
        
        # Проверка наличия и качества стереоизображений
        if not verify_stereo_images():
            self.error = "Проблемы с изображениями"
            return  # Прерывание калибровки при проблемах с изображениями
        
        self.status_message = "Выполнение стереокалибровки..."
        self.progress = 60  # Установка прогресса на 60%
        
        # Вызов функции стереокалибровки
        # 'captures_stereo' - папка с синхронизированными стереоизображениями
        stereo_calibration.calibrate(
            'captures_stereo',      # Путь к папке с изображениями
            'left_*.jpg',           # Шаблон имен для левых изображений
            'right_*.jpg',          # Шаблон имен для правых изображений
            'output',               # Папка для сохранения результатов
            self.params.get('chessboard_size', CHESSBOARD_SIZE),  # Размер шахматной доски
            self.params.get('square_size', SQUARE_SIZE)           # Размер квадрата доски
        )
        
        self.status_message = "Завершение..."
        self.progress = 100  # Установка прогресса на 100%
        
        # Установка флага успешного завершения
        self.result = {'success': True}
    
    def run_universal_calibration(self):
        """
        Универсальная стереокалибровка.
        
        Альтернативный метод калибровки, который может использовать
        разные типы калибровочных паттернов или методы.
        """
        self.status_message = "Выполнение универсальной стереокалибровки..."
        self.progress = 50  # Установка прогресса на 50%
        
        # Вызов функции универсальной стереокалибровки
        universal_stereo_calibration.calibrate(
            'captures',             # Путь к папке с изображениями
            'left_*.jpg',           # Шаблон имен для левых изображений
            'right_*.jpg',          # Шаблон имен для правых изображений
            'output',               # Папка для сохранения результатов
            self.params.get('chessboard_size', CHESSBOARD_SIZE),  # Размер шахматной доски
            self.params.get('square_size', SQUARE_SIZE)           # Размер квадрата доски
        )
        
        self.status_message = "Завершение..."
        self.progress = 100  # Установка прогресса на 100%
        
        # Установка флага успешного завершения
        self.result = {'success': True}
    
    def stop(self):
        """
        Остановка потока калибровки.
        
        Устанавливает флаг is_running в False, что позволяет потоку
        корректно завершиться при следующей проверке.
        """
        self.is_running = False


def draw_progress_screen(screen, message, progress, dots_count=0):
    """
    Отрисовка экрана с индикацией прогресса калибровки.
    
    Параметры:
    -----------
    screen : pygame.Surface
        Поверхность PyGame для отрисовки
    message : str
        Текущее статусное сообщение
    progress : int
        Процент выполнения калибровки (0-100)
    dots_count : int, optional
        Счетчик для анимации точек (по умолчанию 0)
        
    Возвращает:
    -----------
    None
        Функция выполняет отрисовку на переданной поверхности
    """
    # Заполнение экрана черным цветом (очистка)
    screen.fill((0, 0, 0))
    
    # Большой шрифт для заголовка экрана
    title_font = pygame.font.SysFont(None, 48)  # Создание шрифта размером 48
    title_text = title_font.render("ВЫПОЛНЕНИЕ КАЛИБРОВКИ", True, (255, 255, 0))  # Желтый текст
    # Выравнивание заголовка по центру экрана
    screen.blit(title_text, (screen.get_width()//2 - title_text.get_width()//2, 50))
    
    # Шрифт для отображения текущего сообщения
    message_font = pygame.font.SysFont(None, 36)
    # Добавление анимированных точек к сообщению (циклическая анимация)
    dots = "." * (dots_count % 4)  # От 0 до 3 точек
    message_text = message_font.render(f"{message}{dots}", True, (255, 255, 255))  # Белый текст
    # Выравнивание сообщения по центру экрана
    screen.blit(message_text, (screen.get_width()//2 - message_text.get_width()//2, 150))
    
    # Параметры прогресс-бара
    bar_width = 600    # Ширина прогресс-бара в пикселях
    bar_height = 30    # Высота прогресс-бара в пикселях
    bar_x = screen.get_width()//2 - bar_width//2  # X-координата для центрирования
    bar_y = 220        # Y-координата прогресс-бара
    
    # Отрисовка фона прогресс-бара (серый прямоугольник)
    pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
    
    # Отрисовка заполненной части прогресс-бара (зеленый прямоугольник)
    fill_width = int(bar_width * (progress / 100))  # Вычисление ширины заполненной части
    pygame.draw.rect(screen, (0, 200, 0), (bar_x, bar_y, fill_width, bar_height))
    
    # Отображение процента выполнения справа от прогресс-бара
    percent_font = pygame.font.SysFont(None, 28)
    percent_text = percent_font.render(f"{progress}%", True, (255, 255, 255))  # Белый текст
    screen.blit(percent_text, (bar_x + bar_width + 10, bar_y))
    
    # Отображение инструкции для пользователя
    instruction_font = pygame.font.SysFont(None, 24)
    instruction_text = instruction_font.render("Пожалуйста, подождите... Не закрывайте программу.", True, (200, 200, 200))  # Серый текст
    screen.blit(instruction_text, (screen.get_width()//2 - instruction_text.get_width()//2, 280))
    
    # Декоративные элементы для улучшения визуального восприятия
    
    # Анимированный пульсирующий круг (индикатор активности)
    # Размер круга меняется синусоидально от 15 до 25 пикселей
    pulse_size = 20 + 10 * abs((time.time() % 1) - 0.5)  # Формула для пульсации
    # Отрисовка круга с голубой обводкой толщиной 2 пикселя
    pygame.draw.circle(screen, (0, 255, 255), 
                      (screen.get_width()//2, 350),  # Центр круга по центру экрана
                      int(pulse_size), 2)  # Радиус и толщина обводки
    
    # Статичный пояснительный текст
    static_font = pygame.font.SysFont(None, 22)
    static_text = static_font.render("Программа обрабатывает изображения и вычисляет параметры калибровки", True, (150, 150, 150))  # Темно-серый текст
    screen.blit(static_text, (screen.get_width()//2 - static_text.get_width()//2, 400))

def select_cameras_visual_pygame():
    """
    Визуальный выбор двух камер из доступных с использованием PyGame.
    
    Возвращает:
    -----------
    (left_cam_idx, right_cam_idx) : tuple
        Кортеж с индексами выбранных камер (левая, правая)
    """
    # Инициализация PyGame для выбора камер
    pygame.init()
    pygame.camera.init()
    
    # Получение списка доступных камер
    cam_list = pygame.camera.list_cameras()
    
    if len(cam_list) < 2:
        print(f"Ошибка: найдено только {len(cam_list)} камер. Нужно минимум 2.")
        return None, None
    
    print("\n" + "="*60)
    print("ВИЗУАЛЬНЫЙ ВЫБОР КАМЕР")
    print("="*60)
    print(f"Найдено камер: {len(cam_list)}")
    print(f"Список камер: {cam_list}")
    
    # Создаем окно для выбора
    screen_width = 800
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Визуальный выбор камер - Выберите 2 камеры (клавиши 0-9)")
    
    # Открываем все камеры для предпросмотра
    cameras = []
    preview_sizes = []
    
    for cam_id in cam_list:
        try:
            # Пробуем открыть камеру с низким разрешением для предпросмотра
            cam = pygame.camera.Camera(cam_id, (320, 240))
            cam.start()
            time.sleep(0.1)  # Даем камере время на инициализацию
            
            # Проверяем, можно ли получить кадр
            frame = cam.get_image()
            if frame:
                cameras.append(cam)
                preview_sizes.append((320, 240))
                print(f"  Камера {cam_id}: успешно открыта")
            else:
                cam.stop()
        except Exception as e:
            print(f"  Камера {cam_id}: ошибка открытия - {e}")
            if 'cam' in locals():
                try:
                    cam.stop()
                except:
                    pass
    
    if len(cameras) < 2:
        print(f"Ошибка: удалось открыть только {len(cameras)} камер")
        return None, None
    
    # Настройки отображения
    selected_ids = []
    instruction_text = "Выберите две камеры клавишами 0-9. ENTER - подтвердить, ESC - отмена"
    status_text = f"Выбрано камер: {len(selected_ids)}/2"
    
    # Основной цикл выбора
    running = True
    while running:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                selected_ids = None
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    selected_ids = None
                
                elif event.key == pygame.K_RETURN:
                    if len(selected_ids) == 2:
                        running = False
                    else:
                        instruction_text = f"Нужно выбрать 2 камеры! Сейчас выбрано: {len(selected_ids)}"
                
                # Проверяем нажатие цифровых клавиш 0-9
                elif pygame.K_0 <= event.key <= pygame.K_9:
                    cam_idx = event.key - pygame.K_0
                    
                    # Проверяем, существует ли камера с таким индексом
                    if 0 <= cam_idx < len(cam_list):
                        # Проверяем, можно ли открыть эту камеру
                        if cam_idx < len(cameras):
                            if cam_idx in selected_ids:
                                # Убираем из выбранных
                                selected_ids.remove(cam_idx)
                                print(f"Камера {cam_idx} отменена")
                            else:
                                if len(selected_ids) < 2:
                                    selected_ids.append(cam_idx)
                                    print(f"Камера {cam_idx} выбрана")
                                else:
                                    instruction_text = "Уже выбрано 2 камеры! Сначала отмените одну (нажмите ее цифру еще раз)"
                            
                            status_text = f"Выбрано камер: {len(selected_ids)}/2"
                        else:
                            instruction_text = f"Камера {cam_idx} недоступна для предпросмотра"
                    else:
                        instruction_text = f"Нет камеры с индексом {cam_idx}"
        
        # Очищаем экран
        screen.fill((0, 0, 0))
        
        # Отображаем предпросмотр камер
        cols = 2  # Количество столбцов
        preview_width = 320
        preview_height = 240
        margin = 20
        
        for i, cam in enumerate(cameras):
            if i >= 8:  # Ограничиваем показ 8 камерами
                break
            
            try:
                # Получаем кадр
                frame = cam.get_image()
                
                # Позиционируем в сетке
                row = i // cols
                col = i % cols
                
                x = margin + col * (preview_width + margin)
                y = margin + row * (preview_height + margin)
                
                # Отображаем кадр
                screen.blit(frame, (x, y))
                
                # Рисуем рамку
                color = (0, 255, 0) if i in selected_ids else (255, 255, 255)
                pygame.draw.rect(screen, color, (x, y, preview_width, preview_height), 3)
                
                # Добавляем текст с номером камеры
                font = pygame.font.SysFont(None, 36)
                text = font.render(f"Камера {i}", True, (255, 255, 255))
                screen.blit(text, (x + 10, y + 10))
                
                # Инструкция по выбору
                key_font = pygame.font.SysFont(None, 24)
                key_text = key_font.render(f"Нажмите {i}", True, (255, 255, 0))
                screen.blit(key_text, (x + 10, y + preview_height - 30))
                
            except Exception as e:
                # Если не удалось получить кадр, рисуем прямоугольник с ошибкой
                row = i // cols
                col = i % cols
                x = margin + col * (preview_width + margin)
                y = margin + row * (preview_height + margin)
                
                pygame.draw.rect(screen, (100, 0, 0), (x, y, preview_width, preview_height))
                error_font = pygame.font.SysFont(None, 24)
                error_text = error_font.render(f"Камера {i}: ошибка", True, (255, 0, 0))
                screen.blit(error_text, (x + 10, y + preview_height // 2))
        
        # Отображаем инструкции
        font = pygame.font.SysFont(None, 28)
        instruction_surface = font.render(instruction_text, True, (255, 255, 0))
        screen.blit(instruction_surface, (margin, screen_height - 100))
        
        status_surface = font.render(status_text, True, (0, 255, 0))
        screen.blit(status_surface, (margin, screen_height - 70))
        
        # Информация о выбранных камерах
        if selected_ids:
            selected_str = f"Выбраны камеры: {selected_ids}"
            selected_surface = font.render(selected_str, True, (0, 255, 255))
            screen.blit(selected_surface, (margin, screen_height - 40))
        
        pygame.display.flip()
        
        # Ограничиваем частоту кадров
        pygame.time.delay(50)
    
    # Закрываем все камеры предпросмотра
    for cam in cameras:
        try:
            cam.stop()
        except:
            pass
    
    pygame.display.quit()
    
    if selected_ids and len(selected_ids) == 2:
        return selected_ids[0], selected_ids[1]
    else:
        return None, None


def determine_left_right_pygame(cam1_id, cam2_id, cam_list):
    """
    Определение левой и правой камер с использованием PyGame.
    
    Параметры:
    -----------
    cam1_id, cam2_id : int
        Индексы выбранных камер
    cam_list : list
        Список всех доступных камер
        
    Возвращает:
    -----------
    (left_cam_idx, right_cam_idx) : tuple
        Кортеж с индексами левой и правой камер
    """
    print(f"\nОпределение расположения камер {cam1_id} и {cam2_id}...")
    
    # Открываем выбранные камеры
    cam1 = pygame.camera.Camera(cam_list[cam1_id], (640, 480))
    cam2 = pygame.camera.Camera(cam_list[cam2_id], (640, 480))
    
    cam1.start()
    cam2.start()
    time.sleep(0.5)  # Даем время на инициализацию
    
    # Создаем окно
    screen = pygame.display.set_mode((1280, 600))
    pygame.display.set_caption("Определение левой/правой камеры - L: слева ЛЕВАЯ, R: слева ПРАВАЯ, ESC: отмена")
    
    selected_left = None
    selected_right = None
    running = True
    
    while running:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                selected_left = None
                selected_right = None
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    selected_left = None
                    selected_right = None
                
                elif event.key == pygame.K_l:
                    selected_left = cam1_id
                    selected_right = cam2_id
                    running = False
                    print(f"Выбрано: Левая камера = {cam1_id}, Правая камера = {cam2_id}")
                
                elif event.key == pygame.K_r:
                    selected_left = cam2_id
                    selected_right = cam1_id
                    running = False
                    print(f"Выбрано: Левая камера = {cam2_id}, Правая камера = {cam1_id}")
        
        try:
            # Получаем кадры
            frame1 = cam1.get_image()
            frame2 = cam2.get_image()
            
            # Масштабируем для отображения
            display1 = pygame.transform.scale(frame1, (640, 480))
            display2 = pygame.transform.scale(frame2, (640, 480))
            
            # Очищаем экран
            screen.fill((0, 0, 0))
            
            # Отображаем кадры
            screen.blit(display1, (0, 0))
            screen.blit(display2, (640, 0))
            
            # Добавляем подписи
            font = pygame.font.SysFont(None, 36)
            text1 = font.render(f"Камера {cam1_id}", True, (0, 255, 0))
            text2 = font.render(f"Камера {cam2_id}", True, (0, 255, 0))
            screen.blit(text1, (10, 10))
            screen.blit(text2, (650, 10))
            
            # Инструкции
            instruction_font = pygame.font.SysFont(None, 28)
            instruction1 = instruction_font.render("Определите ЛЕВУЮ и ПРАВУЮ камеры:", True, (255, 255, 0))
            instruction2 = instruction_font.render("L - слева ЛЕВАЯ камера", True, (255, 255, 0))
            instruction3 = instruction_font.render("R - слева ПРАВАЯ камера", True, (255, 255, 0))
            instruction4 = instruction_font.render("ESC - отмена", True, (255, 255, 0))
            
            screen.blit(instruction1, (10, 500))
            screen.blit(instruction2, (10, 530))
            screen.blit(instruction3, (10, 560))
            screen.blit(instruction4, (10, 590))
            
            pygame.display.flip()
            
        except Exception as e:
            print(f"Ошибка получения кадра: {e}")
            running = False
        
        # Ограничиваем частоту кадров
        pygame.time.delay(50)
    
    # Закрываем камеры
    cam1.stop()
    cam2.stop()
    pygame.display.quit()
    
    return selected_left, selected_right


def select_cameras_interactive():
    """
    Полный интерактивный процесс выбора камер для калибровки.
    
    Возвращает:
    -----------
    (left_cam_idx, right_cam_idx) : tuple
        Кортеж с индексами левой и правой камер
    """
    print("\n" + "="*70)
    print("ВИЗУАЛЬНЫЙ ВЫБОР КАМЕР ДЛЯ КАЛИБРОВКИ")
    print("="*70)
    
    # Инициализация PyGame
    pygame.init()
    pygame.camera.init()
    
    # Получаем список камер
    cam_list = pygame.camera.list_cameras()
    
    if len(cam_list) < 2:
        print(f"\nНедостаточно камер! Найдено: {len(cam_list)}")
        pygame.quit()
        return None, None
    
    print(f"\nНайдено камер: {len(cam_list)}")
    for i, cam_id in enumerate(cam_list):
        print(f"  Камера {i}: {cam_id}")
    
    # Шаг 1: Визуальный выбор двух камер
    print("\nШАГ 1: Выбор двух камер для стереопары...")
    cam1_idx, cam2_idx = select_cameras_visual_pygame()
    
    if cam1_idx is None or cam2_idx is None:
        print("Выбор камер отменен.")
        pygame.quit()
        return None, None
    
    print(f"\nВыбраны камеры: {cam1_idx} и {cam2_idx}")
    
    # Шаг 2: Определение левой/правой камеры
    print("\nШАГ 2: Определение левой и правой камеры...")
    left_idx, right_idx = determine_left_right_pygame(cam1_idx, cam2_idx, cam_list)
    
    if left_idx is None or right_idx is None:
        print("Определение расположения отменено.")
        pygame.quit()
        return None, None
    
    # Шаг 3: Подтверждение
    print("\n" + "-"*70)
    print("ВЫБОР ЗАВЕРШЕН")
    print(f"  Левая камера: индекс {left_idx} (ID: {cam_list[left_idx]})")
    print(f"  Правая камера: индекс {right_idx} (ID: {cam_list[right_idx]})")
    print("-"*70)
    
    """
    # Предлагаем протестировать
    print("\nХотите протестировать выбранную стереопару?")
    print("1. Да, протестировать (5 секунд)")
    print("2. Нет, сразу начать калибровку")
    
    choice = input("Ваш выбор (1/2): ").strip()
    
    if choice == '1':
        print("\nТестирование стереопары...")
        test_stereo_pair_pygame(left_idx, right_idx, cam_list)
    """
    
    
    pygame.quit()
    return left_idx, right_idx


def test_stereo_pair_pygame(cam1_idx, cam2_idx, cam_list, test_time=5):
    """
    Тестирование выбранной стереопары.
    
    Параметры:
    -----------
    cam1_idx, cam2_idx : int
        Индексы камер
    cam_list : list
        Список камер
    test_time : int
        Время тестирования в секундах
    """
    cam1 = pygame.camera.Camera(cam_list[cam1_idx], (640, 480))
    cam2 = pygame.camera.Camera(cam_list[cam2_idx], (640, 480))
    
    cam1.start()
    cam2.start()
    time.sleep(0.5)
    
    screen = pygame.display.set_mode((1280, 550))
    pygame.display.set_caption("Тестирование стереопары - Q: досрочное завершение")
    
    start_time = time.time()
    frame_count = 0
    
    running = True
    while running and (time.time() - start_time < test_time):
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
        
        try:
            # Получаем кадры
            frame1 = cam1.get_image()
            frame2 = cam2.get_image()
            
            # Масштабируем
            display1 = pygame.transform.scale(frame1, (640, 480))
            display2 = pygame.transform.scale(frame2, (640, 480))
            
            # Очищаем экран
            screen.fill((0, 0, 0))
            
            # Отображаем
            screen.blit(display1, (0, 0))
            screen.blit(display2, (640, 0))
            
            # Информация
            font = pygame.font.SysFont(None, 28)
            info1 = font.render(f"Левая камера (индекс {cam1_idx})", True, (0, 255, 0))
            info2 = font.render(f"Правая камера (индекс {cam2_idx})", True, (0, 255, 0))
            elapsed = time.time() - start_time
            info3 = font.render(f"Время: {elapsed:.1f} сек | Кадры: {frame_count}", True, (255, 255, 0))
            info4 = font.render("Q - досрочное завершение", True, (255, 255, 255))
            
            screen.blit(info1, (10, 490))
            screen.blit(info2, (650, 490))
            screen.blit(info3, (10, 520))
            screen.blit(info4, (650, 520))
            
            pygame.display.flip()
            frame_count += 1
            
        except Exception as e:
            print(f"Ошибка получения кадра: {e}")
            break
        
        pygame.time.delay(30)
    
    cam1.stop()
    cam2.stop()
    pygame.display.quit()
    
    print(f"\nТестирование завершено.")
    print(f"Обработано кадров: {frame_count}")
    print(f"Частота кадров: {frame_count/(time.time() - start_time):.1f} FPS")


def surface_to_nparray(surface):
    """
    Конвертирует pygame.Surface в numpy array
    
    Параметры:
        surface (pygame.Surface): изображение в формате pygame
    
    Возвращает:
        np.array: изображение в формате BGR (для OpenCV)
    
    Примечание:
        - Конвертирует RGB -> BGR для совместимости с OpenCV
        - Форма массива: (высота, ширина, 3 канала)
    """
    img_str = pygame.image.tostring(surface, "RGB")
    w, h = surface.get_size()
    img_np = np.frombuffer(img_str, dtype=np.uint8)
    img_np = img_np.reshape((h, w, 3))
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def nparray_to_surface(img_np):
    """
    Конвертирует numpy array в pygame.Surface
    
    Параметры:
        img_np (np.array): изображение в формате BGR
    
    Возвращает:
        pygame.Surface: изображение в формате RGB для pygame
    
    Примечание:
        - Конвертирует BGR -> RGB для отображения в pygame
        - Транспонирует оси для правильного формата в pygame
    """
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.transpose(img_rgb, (1, 0, 2)))

def load_calibration_data_for_stereo(camera_side, resolution):
    """
    Загружает калибровочные данные и готовит карты для исправления дисторсии
    
    Параметры:
        camera_side (str): идентификатор камеры ('left' или 'right')
        resolution (tuple): разрешение камеры (ширина, высота)
    
    Возвращает:
        tuple: (mapx, mapy, roi, newcameramtx)
            mapx, mapy - карты преобразования для cv2.remap()
            roi - область интереса после исправления (x, y, width, height)
            newcameramtx - новая матрица камеры после оптимизации
    
    Примечание:
        - alpha=0 гарантирует сохранение всех пикселей (включая черные)
        - Это обеспечивает одинаковые ROI для обеих камер
    """
    calib_file = f'output/calibration_data_{camera_side}.pkl'
    
    if not os.path.exists(calib_file):
        print(f"Ошибка: файл калибровки не найден: {calib_file}")
        return None, None, None, None
    
    with open(calib_file, 'rb') as f:
        calib_data = pickle.load(f)
    
    mtx = calib_data['camera_matrix']
    dist = calib_data['distortion_coefficients']
    width, height = resolution
    
    # Используем alpha=0 чтобы получить все пиксели (даже черные)
    # Это гарантирует одинаковые ROI для обеих камер
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (width, height), 1, (width, height))  # alpha=0 вместо 1
    
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (width, height), 5)
    
    return mapx, mapy, roi, newcameramtx

def apply_undistort(img_np, mapx, mapy, roi, target_size=None):
    """
    Универсальная функция коррекции искажений (дисторсии)
    
    Параметры:
        img_np (np.array): исходное изображение в формате BGR
        mapx, mapy (np.array): карты преобразования от cv2.initUndistortRectifyMap
        roi (tuple): область интереса (x, y, width, height)
        target_size (tuple, optional): целевой размер (ширина, высота)
    
    Возвращает:
        np.array: исправленное изображение
    
    Примечание:
        - Если target_size указан, изображение масштабируется до этого размера
        - Если target_size не указан, сохраняется оригинальный размер после ROI
        - Комментированный блок обрезки ROI оставлен для возможной модификации
    """
    if mapx is None or mapy is None:
        return img_np
        
    undistorted = cv2.remap(img_np, mapx, mapy, cv2.INTER_LINEAR)
    
    """
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Обрезаем по ROI
    #x, y, w, h = roi
    #if w > 0 and h > 0:
    #    undistorted = undistorted[y:y+h, x:x+w]
    """
    
    # Масштабируем только если указан target_size
    if target_size is not None and undistorted.shape[:2] != target_size:
        undistorted = cv2.resize(undistorted, target_size)
    
    return undistorted

def ensure_same_size_after_undistort(left_img_np, right_img_np, left_maps, right_maps):
    """
    Гарантирует, что после исправления дисторсии изображения будут одинакового размера
    
    Параметры:
        left_img_np (np.array): левое изображение в формате BGR
        right_img_np (np.array): правое изображение в формате BGR
        left_maps (tuple): карты преобразования для левой камеры (mapx, mapy, roi, newcameramtx)
        right_maps (tuple): карты преобразования для правой камеры (mapx, mapy, roi, newcameramtx)
    
    Возвращает:
        tuple: (left_final, right_final, common_size)
            left_final, right_final - исправленные изображения одинакового размера
            common_size - общий размер (ширина, высота)
    
    Примечание:
        - Использует минимальные размеры по ширине и высоте для обрезки
        - Важно для стереокалибровки - изображения должны быть одинакового размера
    """
    left_mapx, left_mapy, left_roi, _ = left_maps
    right_mapx, right_mapy, right_roi, _ = right_maps
    
    # Применяем undistort без изменения размера
    left_undistorted = apply_undistort(left_img_np, left_mapx, left_mapy, left_roi)
    right_undistorted = apply_undistort(right_img_np, right_mapx, right_mapy, right_roi)
    
    # Находим общий размер (минимальный по ширине и высоте)
    left_h, left_w = left_undistorted.shape[:2]
    right_h, right_w = right_undistorted.shape[:2]
    
    common_width = min(left_w, right_w)
    common_height = min(left_h, right_h)
    
    # Обрезаем оба изображения до общего размера
    left_final = left_undistorted[:common_height, :common_width]
    right_final = right_undistorted[:common_height, :common_width]
    
    return left_final, right_final, (common_width, common_height)

def verify_image_sizes(left_img, right_img, stage_name):
    """
    Проверяет и выводит информацию о размерах изображений
    
    Параметры:
        left_img: левое изображение (может быть np.array или pygame.Surface)
        right_img: правое изображение (может быть np.array или pygame.Surface)
        stage_name (str): название этапа для информационного вывода
    
    Возвращает:
        bool: True если размеры совпадают, False если нет
    
    Примечание:
        - Поддерживает оба типа изображений: numpy array и pygame Surface
        - Выводит подробную информацию в консоль
    """
    left_size = left_img.shape if hasattr(left_img, 'shape') else left_img.get_size()
    right_size = right_img.shape if hasattr(right_img, 'shape') else right_img.get_size()
    
    print(f"\n=== ПРОВЕРКА РАЗМЕРОВ ({stage_name}) ===")
    print(f"Левое изображение: {left_size}")
    print(f"Правое изображение: {right_size}")
    
    if left_size != right_size:
        print("⚠️  ВНИМАНИЕ: Размеры не совпадают!")
        return False
    else:
        print("✅ Размеры совпадают")
        return True

def verify_stereo_images():
    """
    Проверяет, что все стереопары имеют одинаковый размер
    
    Возвращает:
        bool: True если все изображения имеют одинаковый размер, False если есть проблемы
    
    Примечание:
        - Проверяет все изображения в папке captures_stereo
        - Сравнивает размеры левых и правых изображений попарно
        - Проверяет, что все пары имеют одинаковый размер
    """
    left_images = glob.glob('captures_stereo/left_*.jpg')
    right_images = glob.glob('captures_stereo/right_*.jpg')
    
    if len(left_images) != len(right_images):
        print(f"❌ Несовпадающее количество изображений: {len(left_images)} левых vs {len(right_images)} правых")
        return False
    
    if len(left_images) == 0:
        print("❌ Не найдено изображений для стереокалибровки")
        return False
    
    reference_size = None
    all_good = True
    
    for i, (left_path, right_path) in enumerate(zip(sorted(left_images), sorted(right_images))):
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        if left_img is None or right_img is None:
            print(f"❌ Ошибка загрузки: {left_path} или {right_path}")
            all_good = False
            continue
        
        if reference_size is None:
            reference_size = left_img.shape
        
        if left_img.shape != right_img.shape:
            print(f"❌ Размеры не совпадают в паре {i}:")
            print(f"   {left_path}: {left_img.shape}")
            print(f"   {right_path}: {right_img.shape}")
            all_good = False
        
        if left_img.shape != reference_size:
            print(f"❌ Размер отличается от эталона в паре {i}:")
            print(f"   Эталон: {reference_size}")
            print(f"   Фактический: {left_img.shape}")
            all_good = False
    
    if all_good:
        print(f"✅ Все {len(left_images)} стереопар имеют одинаковый размер: {reference_size}")
    else:
        print(f"❌ Обнаружены проблемы с размерами изображений!")
    
    return all_good

def check_calibration_quality(calib_file):
    """
    Проверяет качество калибровки по RMS ошибке
    
    Параметры:
        calib_file (str): путь к файлу с калибровочными данными (.pkl)
    
    Возвращает:
        float: значение RMS ошибки или бесконечность при ошибке
    
    Примечание:
        - RMS ошибка < 0.5: отличное качество
        - RMS ошибка 0.5-1.0: среднее качество
        - RMS ошибка > 1.0: высокое - требуется перекалибровка
        - Выводит рекомендации по качеству калибровки
    """
    try:
        with open(calib_file, 'rb') as f:
            calib_data = pickle.load(f)
        
        rms_error = calib_data.get('reprojection_error', float('inf'))
        print(f"RMS ошибка калибровки: {rms_error:.4f}")
        
        if rms_error > 1.0:
            print("⚠️  ВНИМАНИЕ: Высокая ошибка калибровки!")
            print("   Рекомендуется переделать калибровку с более качественными изображениями")
        elif rms_error > 0.5:
            print("⚠️  Предупреждение: Средняя ошибка калибровки")
        else:
            print("✅ Отличное качество калибровки!")
        
        return rms_error
    except Exception as e:
        print(f"Ошибка при проверке калибровки: {e}")
        return float('inf')

def main():
    """
    Основная функция программы для калибровки стереокамер
    
    Основные функции:
        1. Инициализация двух камер
        2. Определение левой и правой камер
        3. Захват изображений шахматной доски
        4. Индивидуальная калибровка каждой камеры
        5. Стереокалибровка (определение взаимного положения камер)
        6. Управление через графический интерфейс pygame
    
    Управление клавишами:
        - ПРОБЕЛ: сохранить текущие изображения
        - C: выполнить индивидуальную калибровку камер
        - L: загрузить существующие калибровочные данные
        - S: выполнить стереокалибровку
        - U: выполнить универсальную стереокалибровку
        - ESC: выход из программы
    
    Примечание:
        - Для калибровки требуется минимум 10 изображений каждой камеры
        - Шахматная доска должна быть хорошо видна на всех изображениях
        - Изображения сохраняются в папки captures/ и captures_stereo/
    """
    # Инициализация PyGame
    pygame.init()
    pygame.camera.init()
    
    # Выбор камер
    print("\n" + "="*70)
    print("ПРОГРАММА КАЛИБРОВКИ СТЕРЕОКАМЕР")
    print("="*70)
    
    """
    print("\nРЕЖИМЫ ВЫБОРА КАМЕР:")
    print("   [1] Визуальный выбор (рекомендуется)")
    print("   [2] Использовать камеры 0 и 1 по умолчанию")
    print("   [3] Ручной ввод индексов камер")
    
    camera_choice = input("\nВыберите режим (1-3): ").strip()
    """
    
    camera_choice = '1'
    
    if camera_choice == '1':
        # Визуальный выбор камер
        left_idx, right_idx = select_cameras_interactive()
        
        if left_idx is None or right_idx is None:
            print("Не удалось выбрать камеры. Выход.")
            pygame.quit()
            sys.exit(1)
            
    elif camera_choice == '2':
        # Использование камер по умолчанию (0 и 1)
        cam_list = pygame.camera.list_cameras()
        if len(cam_list) < 2:
            print(f"Недостаточно камер! Найдено: {len(cam_list)}")
            pygame.quit()
            sys.exit(1)
        
        left_idx, right_idx = 0, 1
        print(f"Используем камеры по умолчанию: 0 и 1")
        
        # Определение левой/правой
        left_idx, right_idx = determine_left_right_pygame(left_idx, right_idx, cam_list)
        if left_idx is None or right_idx is None:
            print("Определение расположения отменено.")
            pygame.quit()
            sys.exit(1)
            
    elif camera_choice == '3':
        # Ручной ввод
        cam_list = pygame.camera.list_cameras()
        print(f"\nДоступные камеры: {cam_list}")
        print(f"Индексы: 0-{len(cam_list)-1}")
        
        try:
            left_idx = int(input("Введите индекс левой камеры: "))
            right_idx = int(input("Введите индекс правой камеры: "))
            
            if left_idx < 0 or left_idx >= len(cam_list) or right_idx < 0 or right_idx >= len(cam_list):
                print("Ошибка: индекс камеры вне допустимого диапазона!")
                pygame.quit()
                sys.exit(1)
            
            if left_idx == right_idx:
                print("Ошибка: нужно выбрать две разные камеры!")
                pygame.quit()
                sys.exit(1)
            
            # Определение левой/правой
            left_idx, right_idx = determine_left_right_pygame(left_idx, right_idx, cam_list)
            if left_idx is None or right_idx is None:
                print("Определение расположения отменено.")
                pygame.quit()
                sys.exit(1)
                
        except ValueError:
            print("Ошибка: введите целые числа!")
            pygame.quit()
            sys.exit(1)
    else:
        print("Неверный выбор. Используем визуальный выбор.")
        left_idx, right_idx = select_cameras_interactive()
        if left_idx is None or right_idx is None:
            print("Не удалось выбрать камеры. Выход.")
            pygame.quit()
            sys.exit(1)
    
    # После выбора камер переинициализируем PyGame для основной программы
    #print("\nИнициализация PyGame для основной программы...")
    
    # Останавливаем текущие камеры если они были открыты
    try:
        pygame.quit()
    except:
        pass
    
    # Ждем немного для очистки ресурсов
    time.sleep(0.5)
    
    # Переинициализируем PyGame
    pygame.init()
    pygame.camera.init()
    
    # Получаем список камер еще раз
    cam_list = pygame.camera.list_cameras()
    
    # Проверяем, что выбранные камеры все еще доступны
    if left_idx >= len(cam_list) or right_idx >= len(cam_list):
        print(f"Ошибка: одна из выбранных камер больше не доступна!")
        print(f"Доступные камеры: {cam_list}")
        pygame.quit()
        return
    
    print(f"\nИнициализация камер:")
    print(f"  Левая камера: индекс {left_idx}, ID: {cam_list[left_idx]}")
    print(f"  Правая камера: индекс {right_idx}, ID: {cam_list[right_idx]}")

    # Попытка открыть камеры с разными разрешениями
    cameras = []
    original_resolutions = []
    resolutions = [(1920, 1080), (1280, 720), (640, 480)]
    
    for i, cam_idx in enumerate([left_idx, right_idx]):
        cam = None
        original_resolution = None
        cam_id = cam_list[cam_idx]
        
        for res in resolutions:
            cam = pygame.camera.Camera(cam_id, res)
            try:
                cam.start()
                time.sleep(0.1)
                frame = cam.get_image()
                actual_size = frame.get_size()
                if actual_size == res:
                    original_resolution = res
                    print(f"Камера {i} открыта с разрешением {res}")
                    break
                else:
                    print(f"Камера {i}: разрешение {res} не поддерживается (фактический размер {actual_size}), пробую следующее...")
                    cam.stop()
            except Exception as e:
                print(f"Ошибка при попытке открыть камеру {i} с разрешением {res}: {e}")
                if cam:
                    cam.stop()
                cam = None
        
        if not cam or not original_resolution:
            print(f"Не удалось открыть камеру {i} с поддерживаемым разрешением")
            pygame.quit()
            sys.exit(1)
            
        cameras.append(cam)
        original_resolutions.append(original_resolution)

    # Теперь cameras[0] - левая камера, cameras[1] - правая камера
    left_cam, right_cam = cameras[0], cameras[1]
    left_res, right_res = original_resolutions[0], original_resolutions[1]
    
    print(f"Левая камера: разрешение {left_res}")
    print(f"Правая камера: разрешение {right_res}")

    # Создание основного окна
    screen = pygame.display.set_mode((1280, 480))
    pygame.display.set_caption("Двойной видеопоток - ПРОБЕЛ: сохранить | C: калибровка | L: загрузить калибровку | S: стереокалибровка  | U: универсальная калибровка")

    # Счётчик для уникальных номеров
    counter = 0  # счетчик сырых изображений
    stereo_counter = 0  # счетчик исправленных изображений для стереокалибровки
    font = pygame.font.SysFont(None, 24)
    last_save_time = 0
    save_cooldown = 1.0  # Задержка между сохранениями в секундах
    
    # Флаги состояния
    calibration_done = False  # выполнена ли индивидуальная калибровка
    stereo_calibration_done = False  # выполнена ли стереокалибровка
    
    # Переменные для undistort
    left_mapx, left_mapy, left_roi, left_newcameramtx = None, None, None, None
    right_mapx, right_mapy, right_roi, right_newcameramtx = None, None, None, None
    
    # Размер для отображения в реальном времени
    display_size = (640, 480)
    
    # Переменные для управления потоками калибровки
    calibration_thread = None
    dots_counter = 0
    last_dots_update = time.time()
    calibration_result_queue = queue.Queue()

    running = True
    clock = pygame.time.Clock()  # Для контроля FPS
    while running:
        current_time = time.time()
        
        # Обновляем счетчик точек для анимации
        if current_time - last_dots_update > 0.5:
            dots_counter = (dots_counter + 1) % 4
            last_dots_update = current_time
        
        # Проверяем, есть ли результаты из потока калибровки
        if calibration_thread and not calibration_thread.is_alive():
            # Поток завершился, обрабатываем результат
            if calibration_thread.error:
                print(f"❌ Ошибка при калибровке: {calibration_thread.error}")
            elif calibration_thread.result:
                if calibration_thread.calibration_type == 'individual':
                    result = calibration_thread.result
                    if result['success']:
                        # Загружаем калибровочные данные
                        left_mapx, left_mapy, left_roi, left_newcameramtx = load_calibration_data_for_stereo('left', left_res)
                        right_mapx, right_mapy, right_roi, right_newcameramtx = load_calibration_data_for_stereo('right', right_res)
                        
                        if left_mapx is not None and right_mapx is not None:
                            calibration_done = True
                            stereo_counter = 0
                            print("✅ Индивидуальная калибровка завершена!")
                            print("Теперь делайте снимки ИСПРАВЛЕННЫХ изображений для стереокалибровки (ПРОБЕЛ)")
                        else:
                            print("❌ Ошибка загрузки калибровочных данных")
                    else:
                        print("❌ Качество калибровки недостаточно хорошее.")
                
                elif calibration_thread.calibration_type == 'stereo':
                    calibration_done = True
                    stereo_calibration_done = True
                    print("✅ Стереокалибровка завершена успешно!")
                
                elif calibration_thread.calibration_type == 'universal':
                    stereo_calibration_done = True
                    print("✅ Универсальная стереокалибровка завершена успешно!")
            
            calibration_thread = None
        
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and (current_time - last_save_time) > save_cooldown and not calibration_thread:
                    try:
                        # Сохранение изображений при нажатии ПРОБЕЛА
                        left_img_original = left_cam.get_image()
                        right_img_original = right_cam.get_image()
                        
                        # Генерация уникального номера на основе времени
                        unique_id = int(time.time() * 1000)
                        
                        # Создаем папки для сохранения
                        os.makedirs("captures", exist_ok=True)
                        os.makedirs("captures_stereo", exist_ok=True)
                        
                        if not calibration_done:
                            # Сохранение сырых изображений для индивидуальной калибровки
                            pygame.image.save(left_img_original, f"captures/left_{unique_id}.jpg")
                            pygame.image.save(right_img_original, f"captures/right_{unique_id}.jpg")
                            print(f"Сырые сохранено: captures/left_{unique_id}.jpg и captures/right_{unique_id}.jpg")
                            counter += 1
                        else:
                            # Преобразуем в numpy для обработки OpenCV
                            left_img_np = surface_to_nparray(left_img_original)
                            right_img_np = surface_to_nparray(right_img_original)
                            
                            # Применяем undistort с гарантией одинакового размера
                            left_undistorted, right_undistorted, common_size = ensure_same_size_after_undistort(
                                left_img_np, right_img_np, 
                                (left_mapx, left_mapy, left_roi, left_newcameramtx),
                                (right_mapx, right_mapy, right_roi, right_newcameramtx)
                            )
                            
                            # Сохраняем исправленные изображения одинакового размера
                            cv2.imwrite(f"captures_stereo/left_{unique_id}.jpg", left_undistorted)
                            cv2.imwrite(f"captures_stereo/right_{unique_id}.jpg", right_undistorted)
                            
                            print(f"Исправленные сохранено (размер {common_size}): captures_stereo/left_{unique_id}.jpg и captures_stereo/right_{unique_id}.jpg")
                            stereo_counter += 1
                        
                        last_save_time = current_time
                    except Exception as e:
                        print(f"Ошибка при сохранении изображений: {e}")
                
                # Запуск индивидуальной калибровки
                elif event.key == pygame.K_c and not calibration_thread:
                    if counter >= 10:
                        print("Запуск калибровок по отдельности в фоновом потоке...")
                        calibration_thread = CalibrationThread('individual', {
                            'chessboard_size': CHESSBOARD_SIZE,
                            'square_size': SQUARE_SIZE
                        })
                        calibration_thread.start()
                    else:
                        print(f"❌ Недостаточно изображений для калибровки! Нужно 10, получено {counter}")
                
                # Загрузка существующих калибровочных файлов
                elif event.key == pygame.K_l and not calibration_thread:
                    print("Попытка загрузить существующие калибровочные файлы...")
                    
                    # Проверяем качество существующей калибровки
                    print("Проверка качества существующей калибровки:")
                    left_rms = check_calibration_quality('output/calibration_data_left.pkl')
                    right_rms = check_calibration_quality('output/calibration_data_right.pkl')
                    
                    if left_rms < 3.0 and right_rms < 3.0:
                        left_mapx, left_mapy, left_roi, left_newcameramtx = load_calibration_data_for_stereo('left', left_res)
                        right_mapx, right_mapy, right_roi, right_newcameramtx = load_calibration_data_for_stereo('right', right_res)
                        
                        if left_mapx is not None and right_mapx is not None:
                            calibration_done = True
                            stereo_counter = 0
                            print("✅ Существующие калибровочные файлы успешно загружены.")
                            print("Теперь делайте снимки ИСПРАВЛЕННЫХ изображений для стереокалибровки (ПРОБЕЛ)")
                        else:
                            print("❌ Ошибка загрузки калибровочных данных.")
                    else:
                        print("❌ Качество существующей калибровки недостаточно.")
                
                # Запуск стереокалибровки
                elif event.key == pygame.K_s and not calibration_thread:
                    if calibration_done:
                        if stereo_counter >= 10:
                            print("Запуск стереокалибровки в фоновом потоке...")
                            calibration_thread = CalibrationThread('stereo', {
                                'chessboard_size': CHESSBOARD_SIZE,
                                'square_size': SQUARE_SIZE
                            })
                            calibration_thread.start()
                        else:
                            print(f"❌ Недостаточно изображений для стереокалибровки! Нужно 10, получено {stereo_counter}")
                    else:
                        print("❌ Сначала загрузите или выполните индивидуальную калибровку (нажмите L или C)")
                
                # Запуск универсальной стереокалибровки
                elif event.key == pygame.K_u and not calibration_thread:
                    if counter >= 10:
                        print("Запуск универсальной стереокалибровки в фоновом потоке...")
                        calibration_thread = CalibrationThread('universal', {
                            'chessboard_size': CHESSBOARD_SIZE,
                            'square_size': SQUARE_SIZE
                        })
                        calibration_thread.start()
                    else:
                        print(f"❌ Недостаточно изображений для универсальной стереокалибровки! Нужно 10, получено {counter}")
        
        # Отрисовка экрана
        try:
            # Если идет калибровка, показываем экран прогресса
            if calibration_thread and calibration_thread.is_alive():
                draw_progress_screen(screen, calibration_thread.status_message, 
                                   calibration_thread.progress, dots_counter)
            
            else:
                # Нормальный режим - показываем камеры и информацию
                
                # Получение оригинальных кадров
                left_img_original = left_cam.get_image()
                right_img_original = right_cam.get_image()
                
                # Преобразуем в numpy для обработки
                left_img_np = surface_to_nparray(left_img_original)
                right_img_np = surface_to_nparray(right_img_original)
                
                # Размер изображений (опираемся на левое)
                img_size =(left_img_original.get_width() ,left_img_original.get_height())
                
                # Применяем undistort если калибровка выполнена
                if calibration_done:
                    left_undistorted = apply_undistort(left_img_np, left_mapx, left_mapy, left_roi, img_size)
                    right_undistorted = apply_undistort(right_img_np, right_mapx, right_mapy, right_roi, img_size)
                    
                    left_display = pygame.transform.scale(nparray_to_surface(left_undistorted), display_size)
                    right_display = pygame.transform.scale(nparray_to_surface(right_undistorted), display_size)
                else:
                    left_display = pygame.transform.scale(left_img_original, display_size)
                    right_display = pygame.transform.scale(right_img_original, display_size)
                
                screen.blit(left_display, (0, 0))
                screen.blit(right_display, (640, 0))
                
                # Отображение информации
                status = "Режим: Сырые кадры" if not calibration_done else "Режим: Исправленные кадры"
                if stereo_calibration_done:
                    status = "✅ Все калибровки завершены!"
                    
                info_text = f"{status} | Сырые: {counter} | Стерео: {stereo_counter} | Разрешения: {left_res}, {right_res}"
                text_surface = font.render(info_text, True, (0, 255, 0))
                screen.blit(text_surface, (10, 10))
                
                # Дополнительная информация
                if calibration_done and not stereo_calibration_done:
                    help_text2 = "Снимайте исправленные изображения для стереокалибровки!"
                    help_surface2 = font.render(help_text2, True, (255, 100, 100))
                    screen.blit(help_surface2, (10, 40))
                
                # Инструкция
                help_text = "ПРОБЕЛ: сохранить | ESC: выход | C: калибровка | L: загрузить | S: стереокалибровка | U: универсальная калибровка"
                help_surface = font.render(help_text, True, (255, 255, 0))
                screen.blit(help_surface, (10, 450))
                
                # Предупреждение о калибровке
                if calibration_thread:
                    warning_font = pygame.font.SysFont(None, 28)
                    warning_text = warning_font.render("КАЛИБРОВКА ВЫПОЛНЯЕТСЯ...", True, (255, 0, 0))
                    screen.blit(warning_text, (screen.get_width()//2 - warning_text.get_width()//2, 10))
            
            pygame.display.flip()
            
        except Exception as e:
            print(f"Ошибка при отрисовке: {e}")
            running = False
        
        # Ограничиваем FPS для снижения нагрузки на CPU
        clock.tick(30)
    
    # Остановка камер и выход
    left_cam.stop()
    right_cam.stop()
    
    # Останавливаем поток калибровки если он еще работает
    if calibration_thread and calibration_thread.is_alive():
        calibration_thread.stop()
        calibration_thread.join(timeout=2.0)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()