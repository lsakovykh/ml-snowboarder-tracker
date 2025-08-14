import cv2
import os
import numpy as np
import json
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict, Any

def generate_drone_commands(
    object_center: Tuple[int, int], # Координаты центра объекта (cx, cy) в исходном кадре
    object_bbox_size: Tuple[int, int], # Ширина и высота bounding box объекта (width, height) в исходном кадре
    original_frame_width: int, # Ширина исходного видеокадра
    original_frame_height: int, # Высота исходного видеокадра
    config: Dict[str, Any]
) -> Dict[str, str]:
    """
    Генерирует базовые команды для виртуального дрона на основе положения
    и размера отслеживаемого объекта относительно исходного видеокадра.

    Команды предназначены для удержания объекта в центре кадра и на оптимальном расстоянии.

    Args:
        object_center (Tuple[int, int]): Координаты (x, y) центра обнаруженного объекта
                                         в пикселях исходного видеокадра.
        object_bbox_size (Tuple[int, int]): Размеры (ширина, высота) ограничивающей рамки
                                            объекта в пикселях исходного видеокадра.
        original_frame_width (int): Ширина исходного видеокадра в пикселях.
        original_frame_height (int): Высота исходного видеокадра в пикселях.
        config (Dict[str, Any]): Словарь конфигурации, содержащий параметры 'tracking.tolerance_pixels'
                                         и 'tracking.target_bbox_area_percentage' для вычисления команд.

    Returns:
        Dict[str, str]: Словарь с генерируемыми командами для дрона.
                        Ключи: "horizontal" (LEFT/RIGHT/NONE), "vertical" (UP/DOWN/NONE),
                        "distance" (FORWARD/BACKWARD/NONE).
    """
    commands = {
        "horizontal": "NONE",
        "vertical": "NONE",
        "distance": "NONE"
    }
    tolerance_pixels = config['tracking']['tolerance_pixels']
    target_bbox_area_percentage = config['tracking']['target_bbox_area_percentage']

    # Вычисляем центр исходного кадра
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

    # 3. Управление дистанцией
    current_obj_area = obj_width * obj_height
    original_total_frame_area = original_frame_width * original_frame_height
    target_obj_area = original_total_frame_area * target_bbox_area_percentage

    # Пороги для команд "FORWARD" (приблизиться) и "BACKWARD" (отдалиться)
    if current_obj_area < target_obj_area * 0.85:
        commands["distance"] = "FORWARD"
    elif current_obj_area > target_obj_area * 1.15:
        commands["distance"] = "BACKWARD"
    
    return commands


def track_video_and_center_object(
    model_path: str,
    video_input_path: str,
    video_output_path: str,
    config: Dict[str, Any],
) -> None:
    """
    Отслеживает целевой объект (сноубордиста) в видео, используя его уникальный ID,
    и создает новое видео, где объект постоянно центрирован в кадре путем обрезки
    исходного видео. Также логирует данные о трекинге и командах дрона в файл.

    Args:
        model_path (str): Полный или относительный путь к обученной модели YOLO (e.g., 'best.pt').
        video_input_path (str): Полный или относительный путь к исходному видеофайлу.
        video_output_path (str): Полный или относительный путь для сохранения выходного
                                 центрированного видеофайла.
        config (Dict[str, Any]): Словарь конфигурации, содержащий параметры 'tracking' для
                                 определения значений по умолчанию (target_class_id, target_imgsz,
                                 confidence_threshold, iou_threshold).
        target_class_id (int): ID класса объекта, который нужно отслеживать
                               (по умолчанию 0, что соответствует 'snowboarder' в этом проекте).
        target_imgsz (int): Желаемый размер стороны (в пикселях) квадратного выходного
                            видеокадра (например, 640 для 640x640).
        confidence_threshold (float): Порог уверенности (confidence score) для детектирования
                                      объектов. Детекции ниже этого порога отбрасываются.
        iou_threshold (float): Порог Intersection over Union (IoU) для алгоритма NMS (Non-Maximum Suppression).
                               Используется для удаления перекрывающихся ограничивающих рамок.
    """

    # Извлекаем параметры из config
    target_class_id = config['tracking']['target_class_id']
    target_imgsz = config['tracking']['target_imgsz']
    confidence_threshold = config['tracking']['confidence_threshold']
    iou_threshold = config['tracking']['iou_threshold']

    # --- 1. Загрузка модели YOLO ---
    try:
        model = YOLO(model_path)
        print(f"Модель успешно загружена из: '{model_path}'")
    except Exception as e:
        print(f"Критическая ошибка: Не удалось загрузить модель по пути '{model_path}'. Проверьте путь и файл модели. Ошибка: {e}")
        return

    # --- 2. Проверка и создание выходной директории ---
    output_dir = os.path.dirname(video_output_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Выходная директория для видео и логов: '{output_dir}'")

    # --- 3. Чтение входного видео ---
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print(f"Критическая ошибка: Не удалось открыть видеофайл по пути: '{video_input_path}'. Проверьте путь и доступность файла.")
        return

    # Получаем основные свойства исходного видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Исходное видео: '{video_input_path}'")
    print(f"Разрешение: {frame_width}x{frame_height}, FPS: {fps:.2f}, Всего кадров: {total_frames}")

    # --- 4. Подготовка VideoWriter для записи выходного видео ---
    # Список кодеков для попытки инициализации VideoWriter. Порядок важен для совместимости.
    codec_options: List[Tuple[str, str]] = [
        ('mp4v', '.mp4'), # MPEG-4 Video Codec (хорошая совместимость, часто используется)
        ('avc1', '.mp4'), # H.264 (распространенный, но может требовать дополнительных пакетов)
        ('XVID', '.avi'), # XVID MPEG-4 (старый, но универсальный)
        ('MJPG', '.avi')  # Motion JPEG (большой размер файла, но почти всегда поддерживается)
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
                print(f"Успешно инициализирован VideoWriter с кодеком '{codec_fourcc_str}'. Видео будет сохранено как: '{final_video_output_path}'.")
                break
        except Exception as e:
            print(f"Ошибка при попытке инициализации VideoWriter с кодеком '{codec_fourcc_str}': {e}")
        out = None 

    if out is None or not out.isOpened():
        print("Критическая ошибка: Не удалось создать VideoWriter ни с одним из предложенных кодеков. Проверьте установку кодеков, права доступа и работоспособность OpenCV.")
        cap.release()
        return
    
    video_output_path = final_video_output_path
    print(f"Выходное видео будет сохранено в: '{video_output_path}'.")

    # --- 5. Подготовка файла для логирования данных о трекинге ---
    # Файл логов будет создан в той же директории, что и выходное видео.
    log_file_path = os.path.join(os.path.dirname(video_output_path), "tracking_log.jsonl")
    log_file = None
    try:
        log_file = open(log_file_path, 'w', encoding='utf-8')
        print(f"Данные трекинга будут записаны в: '{log_file_path}'.")
    except Exception as e:
        print(f"Критическая ошибка: Не удалось открыть файл логов '{log_file_path}' для записи. Ошибка: {e}")
        cap.release()
        if out: out.release()
        return

    # --- 6. Основной цикл обработки кадров ---
    frame_count = 0
    target_track_id: Optional[int] = None                    # ID целевого сноубордиста, который мы отслеживаем
    last_known_center: Optional[Tuple[int, int]] = None      # Последняя известная позиция центра целевого объекта
    last_known_bbox_size: Optional[Tuple[int, int]] = None   # Последний известный размер bbox целевого объекта
    tracking_status: str = "Инициализация..."                # Текущий статус отслеживания
    
    current_frame_data: Dict[str, Any] = {}                  # Словарь для сбора данных текущего кадра для логирования

    try:
        while True:
            ret, frame = cap.read() # Читаем следующий кадр
            if not ret:
                print(f"Конец видеопотока или ошибка чтения кадра на кадре {frame_count}.")
                break

            frame_count += 1
            if total_frames > 0:
                 print(f"--- Обработано кадров: {frame_count}/{total_frames} ({frame_count / total_frames:.1%}) ---")
            else:
                 print(f"--- Обработано кадров: {frame_count} ---")
            
            # Обнуляем данные для текущего кадра перед обработкой
            current_frame_data = {
                "frame": frame_count,
                "status": "НЕТ",
                "commands": {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"},
                "confidence": None,
                "bbox_size_ratio": None,
                "tracked_id": None
            }

            # 6.1. Выполнение детекции и отслеживания
            results = model.track(frame, persist=True, conf=confidence_threshold, iou=iou_threshold, classes=[target_class_id], verbose=False, tracker='bytetrack.yaml')

            current_target_bbox: Optional[Tuple[int, int, int, int, int, float, int]] = None # (x1, y1, x2, y2, track_id, conf, cls)
            current_target_center: Optional[Tuple[int, int]] = None
            
            # Проверяем, есть ли результаты детекции и трекинга в текущем кадре
            if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None and len(results[0].boxes.id) > 0:
                boxes = results[0].boxes # Доступ к ограничивающим рамкам и ID треков
                
                # Отладочный вывод: сколько объектов найдено
                # print(f"Кадр {frame_count}: Найдено {len(boxes)} объектов класса {target_class_id}.")

                # --- Логика выбора целевого сноубордиста ---
                # Если у нас уже есть ID целевого объекта, пытаемся найти его в текущем кадре
                if target_track_id is not None:
                    found_target_in_current_frame = False
                    for i in range(len(boxes)):
                        # Извлекаем данные детекции/трека
                        track_id = int(boxes.id[i].cpu().item()) # type: ignore
                        if track_id == target_track_id:
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                            conf = float(boxes.conf[i].cpu().item()) # type: ignore
                            cls = int(boxes.cls[i].cpu().item()) # type: ignore
                            current_target_bbox = (x1, y1, x2, y2, track_id, conf, cls)
                            current_target_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            tracking_status = f"Отслеживание ID: {target_track_id}"
                            current_frame_data["confidence"] = conf
                            current_frame_data["tracked_id"] = track_id
                            current_frame_data["status"] = tracking_status
                            found_target_in_current_frame = True
                            break
                    
                    if not found_target_in_current_frame:
                        print(f"Кадр {frame_count}: Целевой сноубордист (ID: {target_track_id}) временно потерян. Поиск...")
                        tracking_status = f"Поиск ID: {target_track_id} (временно потерян)"
                        current_frame_data["status"] = "Потерян"

                # Если целевой ID не задан (это первый кадр, или объект был потерян и нужно выбрать новый), или если мы искали, но не нашли старый ID:
                if current_target_bbox is None:
                    max_area = 0
                    selected_new_target = False
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height

                        # Выбираем самый большой объект в кадре как новую цель
                        if area > max_area:
                            max_area = area
                            track_id = int(boxes.id[i].cpu().item()) # type: ignore
                            conf = float(boxes.conf[i].cpu().item()) # type: ignore
                            cls = int(boxes.cls[i].cpu().item()) # type: ignore
                            current_target_bbox = (x1, y1, x2, y2, track_id, conf, cls)
                            current_target_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            selected_new_target = True
                    
                    if selected_new_target and target_track_id is None:
                        target_track_id = current_target_bbox[4] # type: ignore
                        print(f"Кадр {frame_count}: Инициализация трекинга. Выбран целевой сноубордист с ID {target_track_id}.")
                        tracking_status = f"Выбран ID: {target_track_id}"
                        current_frame_data["confidence"] = current_target_bbox[5] # type: ignore
                        current_frame_data["tracked_id"] = current_target_bbox[4] # type: ignore
                        current_frame_data["status"] = tracking_status

                    elif selected_new_target and target_track_id != current_target_bbox[4]: # type: ignore
                        print(f"Кадр {frame_count}: Целевой сноубордист (ID: {target_track_id}) не найден. Переключено на новый крупнейший объект с ID {current_target_bbox[4]}.") # type: ignore
                        tracking_status = f"Переключено на ID: {current_target_bbox[4]}" # type: ignore
                        target_track_id = current_target_bbox[4] # Обновляем целевой ID # type: ignore 
                        current_frame_data["confidence"] = current_target_bbox[5] # type: ignore
                        current_frame_data["tracked_id"] = current_target_bbox[4] # type: ignore
                        current_frame_data["status"] = tracking_status
                    elif current_target_bbox is None:
                        print(f"Кадр {frame_count}: Сноубордист не найден в этом кадре. Статус: Поиск объекта.")
                        tracking_status = "Поиск объекта..."
                        current_frame_data["status"] = "Поиск"
            else:
                # Если det_results пустой или нет box.id
                print(f"Кадр {frame_count}: Детекций не найдено или трекер не вернул ID. Статус: Поиск объекта.")
                tracking_status = "Поиск объекта..."
                current_frame_data["status"] = "Поиск"

            # 6.2. Генерация команд для дрона и обновление данных для логирования
            drone_commands: Dict[str, str] = {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"}

            if current_target_center is not None and current_target_bbox is not None:
                last_known_center = current_target_center # Обновляем последнюю известную позицию
                x1_bb, y1_bb, x2_bb, y2_bb = current_target_bbox[:4] # type: ignore
                last_known_bbox_size = (int(x2_bb - x1_bb), int(y2_bb - y1_bb))

                # Генерируем команды для дрона на основе положения объекта в исходном кадре
                drone_commands = generate_drone_commands(
                    object_center=last_known_center,
                    object_bbox_size=last_known_bbox_size,
                    original_frame_width=frame_width,
                    original_frame_height=frame_height,
                    config=config
                )
                # print(f"Кадр {frame_count}: Команды дрону: {drone_commands}") # Отключено для уменьшения verbosity
                current_frame_data["commands"] = drone_commands

                # Для расчета bbox_size_ratio также используем размеры исходного кадра
                current_obj_area = last_known_bbox_size[0] * last_known_bbox_size[1]
                original_total_frame_area = frame_width * frame_height
                target_obj_area = original_total_frame_area * config['tracking']['target_bbox_area_percentage']
                
                # Добавлена проверка на target_obj_area > 0 для избежания ZeroDivisionError
                if target_obj_area > 0:
                    current_frame_data["bbox_size_ratio"] = f"{(current_obj_area / target_obj_area * 100):.1f}%"
                else:
                    current_frame_data["bbox_size_ratio"] = "N/A" # Если целевая площадь 0, то N/A
                current_frame_data["status"] = tracking_status


            # 6.3. Запись данных кадра в файл логов (JSONL формат)
            json.dump(current_frame_data, log_file)
            log_file.write('\n') # Каждая запись на новой строке для JSONL

            # 6.4. Формирование и запись центрированного кадра в выходное видео
            if last_known_center is None:
                # print(f"Кадр {frame_count}: Сноубордист не найден ни разу. Запись черного кадра.")
                out.write(np.zeros((target_imgsz, target_imgsz, 3), dtype=np.uint8))
                continue

            # Вычисление области обрезки для центрирования объекта
            cx, cy = last_known_center
            
            # Координаты левого верхнего и правого нижнего углов квадратной области обрезки
            x1_crop = int(cx - target_imgsz / 2)
            y1_crop = int(cy - target_imgsz / 2)
            x2_crop = int(cx + target_imgsz / 2)
            y2_crop = int(cy + target_imgsz / 2)

            # Создаем пустой черный кадр целевого размера для обрезки/вставки
            cropped_frame = np.zeros((target_imgsz, target_imgsz, 3), dtype=np.uint8)
            
            # Координаты для вставки обрезанной секции в черный кадр (с учетом padding)
            # Если x1_crop < 0, то paste_x1 будет > 0, создавая padding слева
            paste_x1 = max(0, -x1_crop)
            paste_y1 = max(0, -y1_crop)
            
            # Координаты для вырезания из исходного кадра (не выходим за границы)
            src_x1 = max(0, x1_crop)
            src_y1 = max(0, y1_crop)
            src_x2 = min(frame_width, x2_crop)
            src_y2 = min(frame_height, y2_crop)

            # Фактические размеры области, которую мы можем вырезать
            actual_crop_width = src_x2 - src_x1
            actual_crop_height = src_y2 - src_y1

            # Вырезаем и вставляем секцию, если она валидна
            if actual_crop_width > 0 and actual_crop_height > 0:
                cropped_section = frame[src_y1:src_y2, src_x1:src_x2]
                
                # Проверяем, что обрезанная секция имеет ожидаемые размеры
                if cropped_section.shape[0] == actual_crop_height and cropped_section.shape[1] == actual_crop_width:
                    cropped_frame[paste_y1 : paste_y1 + actual_crop_height, 
                                  paste_x1 : paste_x1 + actual_crop_width] = cropped_section
                else:
                    print(f"Внимание (Кадр {frame_count}): Несоответствие размеров обрезанной секции ({cropped_section.shape}) и ожидаемых ({actual_crop_height}x{actual_crop_width}). Запись черного кадра.")
                    cropped_frame = np.zeros((target_imgsz, target_imgsz, 3), dtype=np.uint8)
            else:
                # Если нет валидной области для обрезки (например, объект далеко за кадром)
                # print(f"Кадр {frame_count}: Нет валидной области для обрезки или область нулевая. Запись черного кадра.")
                cropped_frame = np.zeros((target_imgsz, target_imgsz, 3), dtype=np.uint8)
            
            # 6.5. Визуализация bounding box и ID на обрезанном кадре
            if current_target_bbox is not None:
                # Извлекаем данные из bbox
                x1, y1, x2, y2, track_id, conf, cls = current_target_bbox
                
                # Преобразуем координаты bbox к относительным в обрезанном кадре
                bbox_x1_rel = int(x1 - x1_crop + paste_x1)
                bbox_y1_rel = int(y1 - y1_crop + paste_y1)
                bbox_x2_rel = int(x2 - x1_crop + paste_x1)
                bbox_y2_rel = int(y2 - y1_crop + paste_y1)
                
                # Убеждаемся, что координаты bbox находятся в пределах целевого размера кадра
                bbox_x1_rel = max(0, bbox_x1_rel)
                bbox_y1_rel = max(0, bbox_y1_rel)
                bbox_x2_rel = min(target_imgsz - 1, bbox_x2_rel)
                bbox_y2_rel = min(target_imgsz - 1, bbox_y2_rel)

                # Рисуем прямоугольник и ID, если bounding box валиден
                if bbox_x2_rel > bbox_x1_rel and bbox_y2_rel > bbox_y1_rel:
                    cv2.rectangle(cropped_frame, (bbox_x1_rel, bbox_y1_rel), (bbox_x2_rel, bbox_y2_rel), (0, 255, 0), 2) # Зеленый BBox
                    text = f"ID: {int(track_id)}" if track_id is not None else "No ID"
                    text_pos_y = max(10, bbox_y1_rel - 10) # Позиция текста: над BBox или вверху кадра
                    cv2.putText(cropped_frame, text, (bbox_x1_rel, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                
            # Записываем финальный обрезанный кадр в выходное видео
            out.write(cropped_frame)

    except Exception as main_e:
        print(f"Критическая ошибка в основном цикле обработки кадров на кадре {frame_count}: {main_e}")

    finally:
        # --- 7. Освобождение ресурсов ---
        print("\nЗавершение обработки: Освобождение ресурсов...")
        if cap is not None:
            cap.release()
            print("Video capture (input) released.")
        if out is not None:
            out.release()
            print("Video writer (output) released.")
        if log_file is not None:
            log_file.close()
            print("Tracking log file closed.")
        cv2.destroyAllWindows()

        print(f"Обработка видео завершена. Результат сохранен в '{video_output_path}'.")
        print(f"Данные трекинга сохранены в '{log_file_path}'.")

