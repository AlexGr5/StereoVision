"""
https://pytorch.org/hub/intelisl_midas_v2/
"""


import torch
import cv2
import numpy as np

# Загрузка модели (выбираем быструю версию для реального времени)
MODEL_TYPE = "MiDaS_small"

# Инициализация модели
midas = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device).eval()

# Настройка трансформаций
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform  # Используем упрощенные преобразования

# Инициализация веб-камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Не удалось подключиться к веб-камере")

print("Нажмите 'q' для выхода")

try:
    while True:
        # Захват кадра с камеры
        ret, frame = cap.read()
        if not ret:
            break
        
        # Подготовка изображения для модели
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)

        # Расчет глубины
        with torch.no_grad():
            prediction = midas(input_batch)
            # Интерполяция до размера исходного кадра
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

        # Нормализация и инверсия (ближние объекты = темные)
        depth_map = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = 255 - depth_map.astype(np.uint8)
        
        # Применение цветовой карты
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

        # Отображение результатов
        cv2.imshow('Оригинал', frame)
        cv2.imshow('Карта глубины', depth_colored)

        # Выход при нажатии 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Очистка ресурсов
    cap.release()

    cv2.destroyAllWindows()
