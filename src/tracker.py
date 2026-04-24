from __future__ import annotations

import numpy as np
import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict, Any, TextIO, Union

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

if TYPE_CHECKING:
    import cv2 as cv2_types
    from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: Union[str, Path]) -> Path:
    """Resolve relative paths from the repository root, not from the current shell."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _require_cv2():
    if cv2 is None:
        raise ImportError("opencv-python не установлен. Установите зависимости: pip install -r requirements.txt")
    return cv2


def _load_yolo_class():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics не установлен. Установите зависимости: pip install -r requirements.txt") from exc
    return YOLO

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
    if obj_width <= 0 or obj_height <= 0:
        return {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"}

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
    if current_obj_area < target_obj_area * 0.9:
        commands["distance"] = "FORWARD"
    elif current_obj_area > target_obj_area * 1.1:
        commands["distance"] = "BACKWARD"
    
    return commands

class TrackerState:
    def __init__(self):
        self.target_track_id: Optional[int] = None
        self.last_known_center: Optional[Tuple[int, int]] = None
        self.last_known_bbox_size: Optional[Tuple[int, int]] = None
        self.tracking_status: str = "Инициализация..."
        self.current_target_bbox: Optional[Tuple[int, int, int, int, int, float, int]] = None
        self.current_target_center: Optional[Tuple[int, int]] = None

def load_model_and_video(model_path: Union[str, Path], video_input_path: Union[str, Path], config: Dict[str, Any]) -> Tuple["YOLO", "cv2_types.VideoCapture"]:
    """
    Загружает модель YOLO и открывает входное видео.

    Args:
        model_path (str): Путь к модели YOLO.
        video_input_path (str): Путь к входному видеофайлу.
        config (Dict[str, Any]): Конфигурация с параметрами.

    Returns:
        Tuple[YOLO, cv2.VideoCapture]: Загруженная модель и объект VideoCapture.
    """
    model_path = resolve_project_path(model_path)
    video_input_path = resolve_project_path(video_input_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Файл модели не найден: '{model_path}'")
    if not video_input_path.is_file():
        raise FileNotFoundError(f"Видеофайл не найден: '{video_input_path}'")

    cv2_module = _require_cv2()
    YOLO = _load_yolo_class()
    try:
        model = YOLO(str(model_path))
        print(f"Модель успешно загружена из: '{model_path}'")
    except Exception as e:
        print(f"Критическая ошибка: Не удалось загрузить модель по пути '{model_path}'. Ошибка: {e}")
        raise

    cap = cv2_module.VideoCapture(str(video_input_path))
    if not cap.isOpened():
        raise OSError(f"Не удалось открыть видеофайл: '{video_input_path}'")
    return model, cap

def setup_output(video_output_path: Union[str, Path], cap: "cv2_types.VideoCapture", config: Dict[str, Any]) -> Tuple["cv2_types.VideoWriter", Path, Path]:
    """
    Настраивает выходной VideoWriter и файл логов.

    Args:
        video_output_path (str): Путь для сохранения выходного видео.
        cap (cv2.VideoCapture): Объект входного видео.
        config (Dict[str, Any]): Конфигурация с параметрами.

    Returns:
        Tuple[cv2.VideoWriter, str, str]: VideoWriter, финальный путь видео, путь лога.
    """
    # Проверка и создание выходной директории
    cv2_module = _require_cv2()
    video_output_path = resolve_project_path(video_output_path)
    output_dir = video_output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Выходная директория для видео и логов: '{output_dir}'")

    # Получаем основные свойства исходного видео
    frame_width = int(cap.get(cv2_module.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2_module.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2_module.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT))

    print(f"Исходное видео: '{video_output_path}'")
    print(f"Разрешение: {frame_width}x{frame_height}, FPS: {fps:.2f}, Всего кадров: {total_frames}")

    # Список кодеков для попытки инициализации VideoWriter. Порядок важен для совместимости.
    codec_options: List[Tuple[str, str]] = [
        ('mp4v', '.mp4'), # MPEG-4 Video Codec (хорошая совместимость, часто используется)
        ('avc1', '.mp4'), # H.264 (распространенный, но может требовать дополнительных пакетов)
        ('XVID', '.avi'), # XVID MPEG-4 (старый, но универсальный)
        ('MJPG', '.avi')  # Motion JPEG (большой размер файла, но почти всегда поддерживается)
    ]

    out = None
    final_video_output_path = video_output_path

    # Попытка инициализации VideoWriter с разными кодеками 
    for codec_fourcc_str, ext in codec_options:
        temp_video_output_path = video_output_path.with_suffix(ext)
        try:
            print(f"Попытка инициализации VideoWriter с кодеком '{codec_fourcc_str}' и расширением '{ext}'...")
            fourcc = cv2_module.VideoWriter_fourcc(*codec_fourcc_str) # type: ignore
            out = cv2_module.VideoWriter(str(temp_video_output_path), fourcc, fps, (config['tracking']['target_imgsz'], config['tracking']['target_imgsz']))
            if out.isOpened():
                final_video_output_path = temp_video_output_path
                print(f"Успешно инициализирован VideoWriter с кодеком '{codec_fourcc_str}'.")
                break
        except Exception as e:
            print(f"Ошибка при попытке инициализации VideoWriter с кодеком '{codec_fourcc_str}': {e}")
        out = None
    if out is None or not out.isOpened():
        raise OSError(f"Не удалось создать VideoWriter для '{video_output_path}'")
    # Файл логов будет создан в той же директории, что и выходное видео
    log_file_path = final_video_output_path.parent / "tracking_log.jsonl"
    return out, final_video_output_path, log_file_path

def process_single_frame(frame: np.ndarray, model: "YOLO", config: Dict[str, Any], state: TrackerState, frame_count: int, frame_width: int, frame_height: int) -> Dict[str, Any]:
    """
    Обрабатывает один кадр: детекция, трекинг, генерация команд.

    Args:
        frame (np.ndarray): Текущий кадр видео.
        model (YOLO): Модель для детекции.
        config (Dict[str, Any]): Конфигурация.
        state (TrackerState): Состояние трекинга.
        frame_count (int): Номер кадра.
        frame_width (int): Ширина кадра.
        frame_height (int): Высота кадра.

    Returns:
        Dict[str, Any]: Данные кадра для логирования.
    """
    current_frame_data = {
        "frame": frame_count,
        "status": "НЕТ",
        "commands": {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"},
        "confidence": None,
        "bbox_size_ratio": None,
        "tracked_id": None
    }

    # Выполнение детекции и отслеживания
    results = model.track(frame, persist=True, conf=config['tracking']['confidence_threshold'], 
                          iou=config['tracking']['iou_threshold'], classes=[config['tracking']['target_class_id']], 
                          verbose=False, tracker=f'{config["tracking"]["tracker_type"]}.yaml')

    state.current_target_bbox = None # (x1, y1, x2, y2, track_id, conf, cls)
    state.current_target_center = None

    # Проверяем, есть ли результаты детекции и трекинга в текущем кадре
    if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None and len(results[0].boxes.id) > 0:
        boxes = results[0].boxes

        # Отладочный вывод: сколько объектов найдено
        # print(f"Кадр {frame_count}: Найдено {len(boxes)} объектов класса {target_class_id}.")

        # --- Логика выбора целевого сноубордиста ---
        # Если у нас уже есть ID целевого объекта, пытаемся найти его в текущем кадре
        if state.target_track_id is not None:
            found_target = False
            for i in range(len(boxes)):
                # Извлекаем данные детекции/трека
                track_id = int(boxes.id[i].cpu().item()) # type: ignore
                if track_id == state.target_track_id:
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().item())
                    cls = int(boxes.cls[i].cpu().item())
                    state.current_target_bbox = (x1, y1, x2, y2, track_id, conf, cls)
                    state.current_target_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    state.tracking_status = f"Отслеживание ID: {state.target_track_id}"
                    current_frame_data["confidence"] = conf
                    current_frame_data["tracked_id"] = track_id
                    current_frame_data["status"] = state.tracking_status
                    found_target = True
                    break
            if not found_target:
                print(f"Кадр {frame_count}: Целевой сноубордист (ID: {state.target_track_id}) временно потерян. Поиск...")
                state.tracking_status = f"Поиск ID: {state.target_track_id} (временно потерян)"
                current_frame_data["status"] = "Потерян"

        # Если целевой ID не задан, выбираем самый крупный объект.
        # Повторный выбор после потери ID отключен по умолчанию, чтобы не перескочить на другого райдера.
        if state.current_target_bbox is None:
            allow_reassignment = config['tracking'].get('allow_target_reassignment', False)
            if state.target_track_id is not None and not allow_reassignment:
                return current_frame_data

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
                    conf = float(boxes.conf[i].cpu().item())
                    cls = int(boxes.cls[i].cpu().item())
                    state.current_target_bbox = (x1, y1, x2, y2, track_id, conf, cls)
                    state.current_target_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    selected_new_target = True
            if selected_new_target and state.target_track_id is None:
                state.target_track_id = state.current_target_bbox[4] # type: ignore
                print(f"Кадр {frame_count}: Инициализация трекинга. Выбран целевой сноубордист с ID {state.target_track_id}.")
                state.tracking_status = f"Выбран ID: {state.target_track_id}"
                current_frame_data["confidence"] = state.current_target_bbox[5] # type: ignore
                current_frame_data["tracked_id"] = state.current_target_bbox[4] # type: ignore
                current_frame_data["status"] = state.tracking_status
            elif selected_new_target and state.target_track_id != state.current_target_bbox[4]: # type: ignore
                print(f"Кадр {frame_count}: Целевой сноубордист (ID: {state.target_track_id}) не найден. Переключено на новый крупнейший объект с ID {state.current_target_bbox[4]}.") # type: ignore
                state.tracking_status = f"Переключено на ID: {state.current_target_bbox[4]}" # type: ignore
                state.target_track_id = state.current_target_bbox[4] # type: ignore
                current_frame_data["confidence"] = state.current_target_bbox[5] # type: ignore
                current_frame_data["tracked_id"] = state.current_target_bbox[4] # type: ignore
                current_frame_data["status"] = state.tracking_status
            elif state.current_target_bbox is None:
                print(f"Кадр {frame_count}: Сноубордист не найден в этом кадре. Статус: Поиск объекта.")
                state.tracking_status = "Поиск объекта..."
                current_frame_data["status"] = "Поиск"
    else:
        # Если det_results пустой или нет box.id
        print(f"Кадр {frame_count}: Детекций не найдено или трекер не вернул ID. Статус: Поиск объекта.")
        state.tracking_status = "Поиск объекта..."
        current_frame_data["status"] = "Поиск"

    if state.current_target_center and state.current_target_bbox:
        state.last_known_center = state.current_target_center
        x1_bb, y1_bb, x2_bb, y2_bb = state.current_target_bbox[:4]
        state.last_known_bbox_size = (int(x2_bb - x1_bb), int(y2_bb - y1_bb))

        # Генерируем команды для дрона на основе положения объекта в исходном кадре
        drone_commands = generate_drone_commands(
            object_center=state.last_known_center,
            object_bbox_size=state.last_known_bbox_size,
            original_frame_width=frame_width,
            original_frame_height=frame_height,
            config=config
        )

        # print(f"Кадр {frame_count}: Команды дрону: {drone_commands}")
        current_frame_data["commands"] = drone_commands
        current_obj_area = state.last_known_bbox_size[0] * state.last_known_bbox_size[1]
        original_total_frame_area = frame_width * frame_height
        target_obj_area = original_total_frame_area * config['tracking']['target_bbox_area_percentage']
        
        # Проверка на target_obj_area > 0 для избежания ZeroDivisionError
        if target_obj_area > 0:
            current_frame_data["bbox_size_ratio"] = f"{(current_obj_area / target_obj_area * 100):.1f}%"
        else:
            current_frame_data["bbox_size_ratio"] = "N/A"
        current_frame_data["status"] = state.tracking_status

    return current_frame_data

def write_frame_data(frame: np.ndarray, current_frame_data: Dict[str, Any], state: TrackerState,
                    out: "cv2_types.VideoWriter", log_file: TextIO, config: Dict[str, Any], frame_count: int,
                    frame_width: int, frame_height: int) -> None:
    """
    Записывает данные кадра в лог и видео.

    Args:
        frame (np.ndarray): Текущий кадр.
        current_frame_data (Dict[str, Any]): Данные кадра.
        state (TrackerState): Состояние трекинга.
        out (cv2.VideoWriter): Объект для записи видео.
        log_file: Файл для логирования.
        config (Dict[str, Any]): Конфигурация.
    """
    target_imgsz = config['tracking']['target_imgsz']
    if state.last_known_center is None:
        out.write(np.zeros((target_imgsz, target_imgsz, 3), dtype=np.uint8))
        json.dump(current_frame_data, log_file, ensure_ascii=False)
        log_file.write('\n')
        return

    # Вычисление области обрезки для центрирования объекта
    cx, cy = state.last_known_center

    # Координаты левого верхнего и правого нижнего углов квадратной области обрезки
    x1_crop = int(cx - target_imgsz / 2)
    y1_crop = int(cy - target_imgsz / 2)
    x2_crop = int(cx + target_imgsz / 2)
    y2_crop = int(cy + target_imgsz / 2)
    
    # Пустой черный кадр целевого размера для обрезки/вставки
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
        if cropped_section.shape[0] == actual_crop_height and cropped_section.shape[1] == actual_crop_width:
            cropped_frame[paste_y1 : paste_y1 + actual_crop_height, 
                          paste_x1 : paste_x1 + actual_crop_width] = cropped_section
        else:
            print(f"Внимание (Кадр {frame_count}): Несоответствие размеров обрезанной секции.")
            cropped_frame = np.zeros((target_imgsz, target_imgsz, 3), dtype=np.uint8)
    else:
        # print(f"Кадр {frame_count}: Нет валидной области для обрезки или область нулевая. Запись черного кадра.")
        cropped_frame = np.zeros((target_imgsz, target_imgsz, 3), dtype=np.uint8)

    # Визуализация bounding box и ID на обрезанном кадре
    if state.current_target_bbox is not None:
        cv2_module = _require_cv2()
        x1, y1, x2, y2, track_id, _, _ = state.current_target_bbox
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
            cv2_module.rectangle(cropped_frame, (bbox_x1_rel, bbox_y1_rel), (bbox_x2_rel, bbox_y2_rel), (0, 255, 0), 2)
            text = f"ID: {int(track_id)}" if track_id is not None else "No ID"
            text_pos_y = max(10, bbox_y1_rel - 10)
            cv2_module.putText(cropped_frame, text, (bbox_x1_rel, text_pos_y), cv2_module.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2_module.LINE_AA)

    # Запись финального кадра и данных кадра в файл логов (JSONL формат)
    out.write(cropped_frame)
    json.dump(current_frame_data, log_file, ensure_ascii=False)
    log_file.write('\n')

def cleanup_resources(
    cap: Optional["cv2_types.VideoCapture"],
    out: Optional["cv2_types.VideoWriter"],
    log_file: Optional[TextIO],
    video_output_path: Optional[Union[str, Path]],
    log_file_path: Optional[Union[str, Path]],
) -> None:
    """
    Освобождает все ресурсы после обработки.

    Args:
        cap (cv2.VideoCapture): Объект входного видео.
        out (cv2.VideoWriter): Объект выходного видео.
        log_file: Файл логов.
        video_output_path (str): Путь к выходному видео.
        log_file_path (str): Путь к файлу логов.
    """
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
    if cv2 is not None:
        cv2.destroyAllWindows()

    if video_output_path is not None:
        print(f"Обработка видео завершена. Результат сохранен в '{video_output_path}'.")
    if log_file_path is not None:
        print(f"Данные трекинга сохранены в '{log_file_path}'.")

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
                                 определения всех настроек (target_class_id, target_imgsz,
                                 confidence_threshold, iou_threshold).
    """
    state = TrackerState()
    cap: Optional["cv2_types.VideoCapture"] = None
    out: Optional["cv2_types.VideoWriter"] = None
    log_file: Optional[TextIO] = None
    log_file_path: Optional[Path] = None
    final_video_output_path: Optional[Path] = None
    frame_count = 0
    try:
        model, cap = load_model_and_video(model_path, video_input_path, config)
        out, final_video_output_path, log_file_path = setup_output(video_output_path, cap, config)
        log_file = open(log_file_path, 'w', encoding='utf-8')
        print(f"Данные трекинга будут записаны в: '{log_file_path}'.")

        cv2_module = _require_cv2()
        frame_width = int(cap.get(cv2_module.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2_module.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Конец видеопотока или ошибка чтения кадра на кадре {frame_count}.")
                break
            frame_count += 1
            if total_frames > 0:
                print(f"--- Обработано кадров: {frame_count}/{total_frames} ({frame_count / total_frames:.1%}) ---")
            else:
                print(f"--- Обработано кадров: {frame_count} ---")
            current_frame_data = process_single_frame(frame, model, config, state, frame_count, frame_width, frame_height)
            write_frame_data(frame, current_frame_data, state, out, log_file, config, frame_count, frame_width, frame_height)

    except Exception as e:
        print(f"Критическая ошибка в основном цикле обработки кадров на кадре {frame_count}: {e}")
        raise
    finally:
        cleanup_resources(cap, out, log_file, final_video_output_path or video_output_path, log_file_path)

