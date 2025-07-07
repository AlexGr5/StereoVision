import pygame
import pygame.camera
import sys
import time
import os

from camera_calibration import invoke
import viravnivanie_v1_2
import count_rotate_v1

def main():
    # Инициализация Pygame
    pygame.init()
    pygame.camera.init()
    
    # Получение списка доступных камер
    cam_list = pygame.camera.list_cameras()
    print(f"Найдено камер: {len(cam_list)}")
    print("Список идентификаторов камер:", cam_list)
    
    if len(cam_list) < 2:
        print("Ошибка: требуется минимум две камеры")
        if len(cam_list) == 0:
            print("Возможные причины:")
            print("1. Камеры не подключены или не распознаны системой")
            print("2. Драйверы камер не установлены")
            print("3. Программе не предоставлен доступ к камерам")
        elif len(cam_list) == 1:
            print("Найдена только одна камера. Проверьте подключение второй камеры.")
        pygame.quit()
        sys.exit(1)

    # Попытка открыть первую камеру с разными разрешениями
    cam0 = None
    original_resolution0 = None
    resolutions = [(1920, 1080), (1280, 720), (640, 480)]
    for res in resolutions:
        cam0 = pygame.camera.Camera(cam_list[0], res)
        try:
            cam0.start()
            # Ждём немного, чтобы получить кадр
            time.sleep(0.1)
            frame = cam0.get_image()
            actual_size = frame.get_size()
            if actual_size == res:
                original_resolution0 = res
                print(f"Камера 0 открыта с разрешением {res}")
                break
            else:
                print(f"Камера 0: разрешение {res} не поддерживается (фактический размер {actual_size}), пробую следующее...")
                cam0.stop()
        except Exception as e:
            print(f"Ошибка при попытке открыть камеру 0 с разрешением {res}: {e}")
            cam0.stop()
            cam0 = None
    if not cam0 or not original_resolution0:
        print("Не удалось открыть первую камеру с поддерживаемым разрешением")
        pygame.quit()
        sys.exit(1)

    # Попытка открыть вторую камеру с разными разрешениями
    cam1 = None
    original_resolution1 = None
    for res in resolutions:
        cam1 = pygame.camera.Camera(cam_list[1], res)
        try:
            cam1.start()
            time.sleep(0.1)
            frame = cam1.get_image()
            actual_size = frame.get_size()
            if actual_size == res:
                original_resolution1 = res
                print(f"Камера 1 открыта с разрешением {res}")
                break
            else:
                print(f"Камера 1: разрешение {res} не поддерживается (фактический размер {actual_size}), пробую следующее...")
                cam1.stop()
        except Exception as e:
            print(f"Ошибка при попытке открыть камеру 1 с разрешением {res}: {e}")
            cam1.stop()
            cam1 = None
    if not cam1 or not original_resolution1:
        print("Не удалось открыть вторую камеру с поддерживаемым разрешением")
        pygame.quit()
        sys.exit(1)

    # Создание окна
    screen = pygame.display.set_mode((1280, 480))
    pygame.display.set_caption("Двойной видеопоток - ПРОБЕЛ: сохранить снимки | ESC: выход | K - начать калибровку")

    # Счётчик для уникальных номеров
    counter = 0
    font = pygame.font.SysFont(None, 24)
    last_save_time = 0
    save_cooldown = 1.0  # Задержка между сохранениями в секундах

    running = True
    while running:
        current_time = time.time()
        
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and (current_time - last_save_time) > save_cooldown:
                    try:
                        # Сохранение изображений при нажатии ПРОБЕЛА
                        img0_original = cam0.get_image()
                        img1_original = cam1.get_image()
                        
                        # Генерация уникального номера на основе времени
                        unique_id = int(time.time() * 1000)
                        
                        # Создаем папку для сохранения, если её нет
                        os.makedirs("captures", exist_ok=True)
                        
                        # Сохранение изображений
                        pygame.image.save(img0_original, f"captures/left_{unique_id}.jpg")
                        pygame.image.save(img1_original, f"captures/right_{unique_id}.jpg")
                        print(f"Сохранено: captures/left_{unique_id}.jpg и captures/right_{unique_id}.jpg")
                        
                        last_save_time = current_time
                        counter += 1
                    except Exception as e:
                        print(f"Ошибка при сохранении изображений: {e}")
                elif event.key == pygame.K_c:
                    if counter >= 10:
                        print("Вызов калибровок")
                        invoke('captures', 'left_*.jpg', 'output_left', 'left', (7, 7), 2.65)
                        invoke('captures', 'right_*.jpg', 'output_right', 'right', (7, 7), 2.65)
                        viravnivanie_v1_2.main()
                        count_rotate_v1.main()
                    else:
                        print("Недостаточно количества изображений для калибровки!\nНеобходимо минимум 10!")
                        

        try:
            # Получение и отображение кадров
            img0_original = cam0.get_image()
            img1_original = cam1.get_image()
            
            # Масштабирование до 640x480 для отображения
            if img0_original.get_size() != (640, 480):
                img0_display = pygame.transform.scale(img0_original, (640, 480))
            else:
                img0_display = img0_original
                
            if img1_original.get_size() != (640, 480):
                img1_display = pygame.transform.scale(img1_original, (640, 480))
            else:
                img1_display = img1_original
            
            screen.blit(img0_display, (0, 0))
            screen.blit(img1_display, (640, 0))
            
            # Отображение информации
            info_text = f"Камеры: {cam_list[0]} и {cam_list[1]} | Сохранено снимков: {counter} | Разрешения: {original_resolution0}, {original_resolution1}"
            text_surface = font.render(info_text, True, (0, 255, 0))
            screen.blit(text_surface, (10, 10))
            
            # Инструкция
            help_text = "ПРОБЕЛ: сохранить снимки | ESC: выход"
            help_surface = font.render(help_text, True, (255, 255, 0))
            screen.blit(help_surface, (10, 450))
            
            pygame.display.flip()
        except Exception as e:
            print(f"Ошибка при получении кадра: {e}")
            running = False

    # Остановка камер и выход
    cam0.stop()
    cam1.stop()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()