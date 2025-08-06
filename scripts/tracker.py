import cv2
import os
import numpy as np
from ultralytics import YOLO
from typing import Tuple, Optional # Добавлен Optional для более точных типов

def track_video_and_center_object(
    model_path: str,
    video_input_path: str,
    video_output_path: str,
    target_class_id: int = 0, # 0 для класса 'snowboarder' в нашей модели
    target_imgsz: int = 640, # Размер квадратного кадра, который будем вырезать
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.7
) -> None:
    """
    Отслеживает целевой объект в видео и создает новое видео,
    где объект центрирован в кадре путем обрезки.

    Args:
        model_path (str): Путь к обученной модели YOLO.
        video_input_path (str): Путь к исходному видеофайлу.
        video_output_path (str): Путь для сохранения выходного видеофайла.
        target_class_id (int): ID класса отслеживаемого объекта (по умолчанию 0 для 'snowboarder').
        target_imgsz (int): Желаемый размер (сторона квадрата) выходного видеокадра.
        confidence_threshold (float): Порог уверенности для детекции.
        iou_threshold (float): Порог IoU для не-максимального подавления (NMS).
    """

    # 1. Загрузка модели
    try:
        model = YOLO(model_path)
        print(f"Модель успешно загружена из: {model_path}")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    # 2. Проверка и создание выходной директории
    output_dir = os.path.dirname(video_output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана выходная директория: {output_dir}")

    # 3. Чтение видео
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл {video_input_path}")
        return

    # Получаем свойства видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Исходное видео: {video_input_path}")
    print(f"Разрешение: {frame_width}x{frame_height}, FPS: {fps}, Всего кадров: {total_frames}")

    # 4. Подготовка для записи выходного видео
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (target_imgsz, target_imgsz))
    except Exception as e:
        print(f"Критическая ошибка: Не удалось создать VideoWriter для {video_output_path} с кодеком mp4v. Проверьте установку кодеков и права доступа. Ошибка: {e}")
        cap.release()
        return

    if not out.isOpened():
        print("Критическая ошибка: Не удалось создать VideoWriter с кодеком mp4v. Попробуйте другой кодек или проверьте установку.")
        cap.release()
        return
    
    print(f"Выходное видео будет сохранено в: {video_output_path} с разрешением {target_imgsz}x{target_imgsz}")

    # --- Основной цикл обработки кадров ---
    frame_count = 0
    
    # last_known_center теперь хранит последнюю известную позицию объекта
    # Это позволит сглаживать движение, если объект временно пропал
    last_known_center: Optional[Tuple[int, int]] = None
    last_known_bbox_size: Optional[Tuple[int, int]] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Конец видео или ошибка чтения на кадре {frame_count}.")
            break # Конец видео

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"--- Обработано кадров: {frame_count}/{total_frames} ---")
            
        # 1. Выполнение детекции и отслеживания
        results = model.track(frame, persist=True, conf=confidence_threshold, iou=iou_threshold, classes=[target_class_id], verbose=False, tracker='bytetrack.yaml') 

        # 2. Выбор целевого сноубордиста
        current_target_bbox: Optional[Tuple] = None
        current_target_center: Optional[Tuple[int, int]] = None
        
        # Проверка, есть ли какие-либо результаты детекции/отслеживания и есть ли в них Boxes с ID
        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None and len(results[0].boxes.id) > 0:
            boxes = results[0].boxes
            
            # Отладочный вывод: сколько объектов найдено
            print(f"Кадр {frame_count}: Найдено {len(boxes)} объектов класса {target_class_id}.")

            max_area = 0
            
            # Итерируемся по отдельным BoxDetection объектов
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = boxes.conf[i].cpu().item()
                cls = boxes.cls[i].cpu().item()
                track_id = boxes.id[i].cpu().item() if boxes.id is not None else -1

                width = x2 - x1
                height = y2 - y1
                area = width * height

                if area > max_area: # Выбираем объект с наибольшей площадью bbox
                    max_area = area
                    current_target_bbox = (x1, y1, x2, y2, track_id, conf, cls) # Формируем кортеж
                    current_target_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            if current_target_center and current_target_bbox:
                 last_known_center = current_target_center
                 x1_bb, y1_bb, x2_bb, y2_bb = current_target_bbox[:4]
                 last_known_bbox_size = (int(x2_bb - x1_bb), int(y2_bb - y1_bb))

        # Если объект не был найден в текущем кадре, используем последнюю известную позицию
        if last_known_center is None:
            # Если объект никогда не был найден, записываем черный кадр
            print(f"Кадр {frame_count}: Сноубордист не найден ни разу. Запись черного кадра.")
            out.write(np.zeros((target_imgsz, target_imgsz, 3), dtype=np.uint8))
            continue # Переходим к следующему кадру

        # 4. Вычисление области обрезки для центрирования
        # Здесь last_known_center гарантированно не None
        cx, cy = last_known_center 
        
        # Вычисляем углы квадратного кадра
        x1_crop = int(cx - target_imgsz / 2)
        y1_crop = int(cy - target_imgsz / 2)
        x2_crop = int(cx + target_imgsz / 2)
        y2_crop = int(cy + target_imgsz / 2)

        # 5. Обработка границ кадра (padding)
        cropped_frame = np.zeros((target_imgsz, target_imgsz, 3), dtype=np.uint8)
        
        paste_x1 = max(0, -x1_crop)
        paste_y1 = max(0, -y1_crop)
        
        src_x1 = max(0, x1_crop)
        src_y1 = max(0, y1_crop)
        src_x2 = min(frame_width, x2_crop)
        src_y2 = min(frame_height, y2_crop)

        actual_crop_width = src_x2 - src_x1
        actual_crop_height = src_y2 - src_y1

        if actual_crop_width > 0 and actual_crop_height > 0:
            cropped_section = frame[src_y1:src_y2, src_x1:src_x2]
            
            if cropped_section.shape[0] == actual_crop_height and cropped_section.shape[1] == actual_crop_width:
                cropped_frame[paste_y1 : paste_y1 + actual_crop_height, 
                              paste_x1 : paste_x1 + actual_crop_width] = cropped_section
            else:
                print(f"Кадр {frame_count} ОШИБКА РАЗМЕРОВ: cropped_section {cropped_section.shape} vs expected {actual_crop_height}x{actual_crop_width}. Запись черного кадра.")
                cropped_frame = np.zeros((target_imgsz, target_imgsz, 3), dtype=np.uint8)
        else:
            print(f"Кадр {frame_count}: Нет области для обрезки или область нулевая. Запись черного кадра.")
            cropped_frame = np.zeros((target_imgsz, target_imgsz, 3), dtype=np.uint8)
        
        # 6. Визуализация (нарисовать bbox на обрезанном кадре)
        if current_target_bbox is not None: # Только если в текущем кадре был найден сноубордист
            # Распаковка здесь безопасна, так как мы только что проверили current_target_bbox на None
            x1, y1, x2, y2, track_id, conf, cls = current_target_bbox
            bbox_x1_rel = int(x1 - x1_crop + paste_x1)
            bbox_y1_rel = int(y1 - y1_crop + paste_y1)
            bbox_x2_rel = int(x2 - x1_crop + paste_x1)
            bbox_y2_rel = int(y2 - y1_crop + paste_y1)
            
            bbox_x1_rel = max(0, bbox_x1_rel)
            bbox_y1_rel = max(0, bbox_y1_rel)
            bbox_x2_rel = min(target_imgsz - 1, bbox_x2_rel)
            bbox_y2_rel = min(target_imgsz - 1, bbox_y2_rel)

            if bbox_x2_rel > bbox_x1_rel and bbox_y2_rel > bbox_y1_rel:
                cv2.rectangle(cropped_frame, (bbox_x1_rel, bbox_y1_rel), (bbox_x2_rel, bbox_y2_rel), (0, 255, 0), 2)
                text = f"ID: {int(track_id)}" if track_id is not None else "No ID"
                text_pos_y = max(10, bbox_y1_rel - 10)
                cv2.putText(cropped_frame, text, (bbox_x1_rel, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        out.write(cropped_frame)

    # 7. Освобождение ресурсов
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Обработка видео завершена. Результат сохранен в {video_output_path}")

