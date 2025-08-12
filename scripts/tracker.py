import cv2
import os
import numpy as np
import json
from ultralytics import YOLO
from typing import Tuple, Optional, Dict, Any # Добавили Dict и Any для более точных типов

def generate_drone_commands(
    object_center: Tuple[int, int],
    object_bbox_size: Tuple[int, int],
    original_frame_width: int,
    original_frame_height: int,
    target_bbox_area_percentage: float = 0.15, # Целевой размер объекта в процентах от площади кадра
    tolerance_pixels: int = 20 # Допустимое отклонение от центра в пикселях
) -> Dict[str, str]:
    """
    Генерирует базовые команды для дрона на основе положения и размера объекта.

    Args:
        object_center (Tuple[int, int]): Координаты центра объекта (cx, cy).
        object_bbox_size (Tuple[int, int]): Ширина и высота bounding box объекта (width, height).
        original_frame_width (int): Ширина исходного кадра
        original_frame_height (int): Высота исходного кадра
        target_bbox_area_percentage (float): Целевая площадь bounding box объекта
                                             как процент от площади кадра.
        tolerance_pixels (int): Допустимое отклонение центра объекта от центра кадра
                                до генерации команды движения.

    Returns:
        Dict[str, str]: Словарь с командами для дрона (например, {'horizontal': 'LEFT', 'vertical': 'NONE', 'distance': 'FORWARD'}).
    """
    commands = {
        "horizontal": "NONE",
        "vertical": "NONE",
        "distance": "NONE"
    }

    original_frame_center_x = float(original_frame_width / 2)
    original_frame_center_y = float(original_frame_height / 2)

    obj_cx, obj_cy = object_center
    obj_width, obj_height = object_bbox_size

    # 1. Горизонтальное управление
    if obj_cx < original_frame_center_x - tolerance_pixels:
        commands["horizontal"] = "LEFT"
    elif obj_cx > original_frame_center_x + tolerance_pixels:
        commands["horizontal"] = "RIGHT"

    # 2. Вертикальное управление
    if obj_cy < original_frame_center_y - tolerance_pixels:
        commands["vertical"] = "UP"
    elif obj_cy > original_frame_center_y + tolerance_pixels:
        commands["vertical"] = "DOWN"

    # 3. Управление дистанцией (вперед/назад)
    current_obj_area = obj_width * obj_height
    original_total_frame_area = original_frame_width * original_frame_height
    target_obj_area = original_total_frame_area * target_bbox_area_percentage

    if current_obj_area < target_obj_area * 0.85:
        commands["distance"] = "FORWARD"
    elif current_obj_area > target_obj_area * 1.15:
        commands["distance"] = "BACKWARD"
    
    return commands


def track_video_and_center_object(
    model_path: str,
    video_input_path: str,
    video_output_path: str,
    target_class_id: int = 0, # 0 для класса 'snowboarder' в нашей модели
    target_imgsz: int = 640,  # Размер квадратного кадра, который будем вырезать
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.7
) -> None:
    """
    Отслеживает целевой объект в видео, используя его уникальный ID,
    и создает новое видео, где объект центрирован в кадре путем обрезки.

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
    out = None
    codec_options = [
        ('mp4v', '.mp4'),
        ('avc1', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi')
    ]

    final_video_output_path = video_output_path

    for codec_fourcc_str, ext in codec_options:
        temp_video_output_path = video_output_path.rsplit('.', 1)[0] + ext
        try:
            print(f"Попытка инициализации VideoWriter с кодеком '{codec_fourcc_str}' и расширением '{ext}'...")
            fourcc = cv2.VideoWriter_fourcc(*codec_fourcc_str) # type: ignore
            out = cv2.VideoWriter(temp_video_output_path, fourcc, fps, (target_imgsz, target_imgsz))
            if out.isOpened():
                final_video_output_path = temp_video_output_path
                print(f"Успешно инициализирован VideoWriter с кодеком '{codec_fourcc_str}'. Видео будет сохранено как: {final_video_output_path}")
                break
            else:
                print(f"Не удалось инициализировать VideoWriter с кодеком '{codec_fourcc_str}'.")
        except Exception as e:
            print(f"Ошибка при попытке инициализации VideoWriter с кодеком '{codec_fourcc_str}': {e}")
        out = None

    if out is None or not out.isOpened():
        print("Критическая ошибка: Не удалось создать VideoWriter ни с одним из предложенных кодеков. Проверьте установку кодеков и права доступа.")
        cap.release()
        return
    
    video_output_path = final_video_output_path
    print(f"Выходное видео будет сохранено в: {video_output_path} с разрешением {target_imgsz}x{target_imgsz}")

    # --- Подготовка файла для логирования данных о трекинге ---
    # Создаем путь к файлу логов в той же директории, что и видео
    log_file_path = os.path.join(os.path.dirname(video_output_path), "tracking_log.jsonl")
    log_file = open(log_file_path, 'w')
    print(f"Данные трекинга будут записаны в: {log_file_path}")
    # ----------------------------------------------------------

    # --- Основной цикл обработки кадров ---
    frame_count = 0

    target_track_id: Optional[int] = None
    last_known_center: Optional[Tuple[int, int]] = None
    last_known_bbox_size: Optional[Tuple[int, int]] = None
    tracking_status: str = "Инициализация..."
    
    current_frame_data: Dict[str, Any] = {} # Словарь для сбора данных текущего кадра

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Конец видео или ошибка чтения на кадре {frame_count}.")
            break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"--- Обработано кадров: {frame_count}/{total_frames} ---")
        
        # Обнуляем данные для текущего кадра
        current_frame_data = {
            "frame": frame_count,
            "status": "НЕТ",
            "commands": {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"},
            "confidence": None,
            "bbox_size_ratio": None,
            "tracked_id": None
        }

        # 1. Выполнение детекции и отслеживания
        results = model.track(frame, persist=True, conf=confidence_threshold, iou=iou_threshold, classes=[target_class_id], verbose=False, tracker='botsort.yaml')

        current_target_bbox: Optional[Tuple[int, int, int, int, int, float, int]] = None 
        current_target_center: Optional[Tuple[int, int]] = None
        
        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None and len(results[0].boxes.id) > 0:
            boxes = results[0].boxes
            
            # Отладочный вывод: сколько объектов найдено
            print(f"Кадр {frame_count}: Найдено {len(boxes)} объектов класса {target_class_id}.")

            # Если у нас уже есть ID целевого объекта, ищем его в текущем кадре
            if target_track_id is not None:
                found_target_in_current_frame = False
                for i in range(len(boxes)):
                    track_id = int(boxes.id[i].cpu().item()) # type: ignore
                    if track_id == target_track_id:
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        conf = float(boxes.conf[i].cpu().item()) # type: ignore
                        cls = int(boxes.cls[i].cpu().item()) # type: ignore
                        current_target_bbox = (x1, y1, x2, y2, track_id, conf, cls)
                        current_target_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        tracking_status = f"Отслеживание ID: {target_track_id}"
                        current_frame_data["confidence"] = conf # Логируем уверенность
                        current_frame_data["tracked_id"] = track_id # Логируем ID
                        found_target_in_current_frame = True
                        break
                if not found_target_in_current_frame:
                    tracking_status = f"Поиск ID: {target_track_id} (временно потерян)"
                    current_frame_data["status"] = "Потерян"

            # Если целевой ID не задан (это первый кадр) или мы его не нашли в этом кадре
            if current_target_bbox is None:
                max_area = 0
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height

                    if area > max_area:
                        max_area = area
                        track_id = int(boxes.id[i].cpu().item()) # type: ignore
                        conf = float(boxes.conf[i].cpu().item()) # type: ignore
                        cls = int(boxes.cls[i].cpu().item()) # type: ignore
                        current_target_bbox = (x1, y1, x2, y2, track_id, conf, cls)
                        current_target_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                
                if target_track_id is None and current_target_bbox is not None:
                    target_track_id = current_target_bbox[4]
                    print(f"Кадр {frame_count}: Первый кадр. Целевой сноубордист с ID {target_track_id} выбран. BBox: ({current_target_bbox[0]}, {current_target_bbox[1]}, {current_target_bbox[2]}, {current_target_bbox[3]})")
                    tracking_status = f"Выбран ID: {target_track_id}"
                    current_frame_data["confidence"] = current_target_bbox[5] # Логируем уверенность
                    current_frame_data["tracked_id"] = current_target_bbox[4] # Логируем ID
                
                elif current_target_bbox is not None and target_track_id != current_target_bbox[4]:
                     print(f"Кадр {frame_count}: Целевой сноубордист (ID: {target_track_id}) не найден. Выбран самый большой объект с новым ID {current_target_bbox[4]}. BBox: ({current_target_bbox[0]}, {current_target_bbox[1]}, {current_target_bbox[2]}, {current_target_bbox[3]})")
                     tracking_status = f"Переключено на ID: {current_target_bbox[4]}"
                     target_track_id = current_target_bbox[4] # Обновляем целевой ID
                     current_frame_data["confidence"] = current_target_bbox[5] # Логируем уверенность
                     current_frame_data["tracked_id"] = current_target_bbox[4] # Логируем ID
                elif current_target_bbox is None:
                    print(f"Кадр {frame_count}: Сноубордист не найден в этом кадре.")
                    tracking_status = "Поиск объекта..."
                    current_frame_data["status"] = "Поиск"
        else:
            print(f"Кадр {frame_count}: Детекций не найдено или трекер не вернул ID.")
            tracking_status = "Поиск объекта..."
            current_frame_data["status"] = "Поиск"

        drone_commands: Dict[str, str] = {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"}

        if current_target_center is not None and current_target_bbox is not None:
            last_known_center = current_target_center
            x1_bb, y1_bb, x2_bb, y2_bb = current_target_bbox[:4] # type: ignore
            last_known_bbox_size = (int(x2_bb - x1_bb), int(y2_bb - y1_bb))

            # Генерируем команды для дрона
            drone_commands = generate_drone_commands(
                object_center=last_known_center,
                object_bbox_size=last_known_bbox_size,
                original_frame_width=frame_width,
                original_frame_height=frame_height
            )
            print(f"Кадр {frame_count}: Команды дрону: {drone_commands}")
            current_frame_data["commands"] = drone_commands

            # Для расчета bbox_size_ratio также используем размеры исходного кадра
            current_obj_area = last_known_bbox_size[0] * last_known_bbox_size[1]
            original_total_frame_area = frame_width * frame_height # <-- ИСПРАВЛЕНО
            target_obj_area = original_total_frame_area * 0.15
            current_frame_data["bbox_size_ratio"] = f"{(current_obj_area / target_obj_area * 100):.1f}%"
            current_frame_data["status"] = tracking_status

        # Запись данных кадра в файл
        json.dump(current_frame_data, log_file)
        log_file.write('\n') # Каждая запись на новой строке

        if last_known_center is None:
            print(f"Кадр {frame_count}: Сноубордист не найден ни разу. Запись черного кадра.")
            out.write(np.zeros((target_imgsz, target_imgsz, 3), dtype=np.uint8))
            continue

        # 3. Вычисление области обрезки для центрирования
        cx, cy = last_known_center 
        
        x1_crop = int(cx - target_imgsz / 2)
        y1_crop = int(cy - target_imgsz / 2)
        x2_crop = int(cx + target_imgsz / 2)
        y2_crop = int(cy + target_imgsz / 2)

        # 4. Обработка границ кадра (padding)
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
        
        # 5. Визуализация оверлеев на обработанном видео
        if current_target_bbox is not None:
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
                cv2.putText(cropped_frame, text, (bbox_x1_rel, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            
        out.write(cropped_frame)

    # 6. Освобождение ресурсов
    cap.release()
    out.release()
    log_file.close() # Закрываем файл логов
    cv2.destroyAllWindows()
    print(f"Обработка видео завершена. Результат сохранен в {video_output_path}")
    print(f"Данные трекинга сохранены в {log_file_path}")

