import cv2
import os
import random
import base64
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from IPython.display import display, Image, HTML, Markdown
import json
from PIL import ImageDraw, ImageFont
from PIL import Image as PILImage

# Попытка импорта wandb. Если не установлен, устанавливаем wandb = None.
try:
    import wandb
except ImportError:
    wandb = None # W&B не установлен или недоступен

# Определяем корневую директорию проекта, исходя из расположения текущего скрипта.
# Это позволяет функциям корректно работать с путями относительно корня проекта,
# независимо от того, откуда запущен Jupyter Notebook или скрипт.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))


def display_image_inline(image_path: str, width: int = 600) -> None:
    """
    Отображает изображение из файла в Jupyter Notebook.

    Args:
        image_path (str): Полный или относительный путь к файлу изображения.
        width (int): Желаемая ширина отображаемого изображения в пикселях.
    """
    try:
        # Проверяем существование файла перед попыткой отображения
        if not os.path.exists(image_path):
            print(f"Ошибка: Файл изображения не найден по пути: {image_path}")
            return
        display(Image(filename=image_path, width=width))
    except Exception as e:
        print(f"Ошибка при отображении изображения '{image_path}': {e}")


def plot_bboxes_on_image(
    image_path: str,
    labels_path: str,
    class_names: Dict[int, str],
    output_dir: Optional[str] = None,
    display_inline: bool = True
) -> None:
    """
    Рисует ограничивающие рамки (Bounding Boxes) на изображении на основе YOLO-аннотаций.
    Отображает изображение в Jupyter Notebook и опционально сохраняет его в файл.

    Args:
        image_path (str): Полный или относительный путь к файлу изображения.
        labels_path (str): Полный или относительный путь к файлу аннотаций YOLO (.txt)
                           для соответствующего изображения.
        class_names (Dict[int, str]): Словарь, сопоставляющий ID класса с его именем
                                     (например, {0: 'snowboarder'}).
        output_dir (Optional[str]): Директория для сохранения изображения с BBoxes.
                                    Если None, изображение не сохраняется.
        display_inline (bool): Если True, отображает изображение в Jupyter Notebook.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Ошибка: Не удалось загрузить изображение по пути {image_path}. Возможно, файл поврежден или путь неверен.")
            return

        h, w, _ = img.shape

        labels: List[str] = []
        if not os.path.exists(labels_path):
            print(f"Внимание: Файл аннотаций не найден для '{os.path.basename(image_path)}' по пути '{labels_path}'. Отображаем изображение без BBoxes.")
        else:
            with open(labels_path, 'r') as f:
                labels = f.readlines()

        for label_line in labels:
            try:
                parts = list(map(float, label_line.strip().split()))
                if len(parts) < 5: # Проверка на минимальное количество частей
                    print(f"Внимание: Некорректная строка аннотации '{label_line.strip()}' в файле '{labels_path}'. Пропускаем.")
                    continue

                class_id = int(parts[0])
                x_center, y_center, bbox_width, bbox_height = parts[1:]

                # Преобразование относительных координат в абсолютные пиксели
                x1 = int((x_center - bbox_width / 2) * w)
                y1 = int((y_center - bbox_height / 2) * h)
                x2 = int((x_center + bbox_width / 2) * w)
                y2 = int((y_center + bbox_height / 2) * h)

                # Отрисовка BBox (зеленый цвет)
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Добавление метки класса
                class_name = class_names.get(class_id, f"Class {class_id}")
                text = f"{class_name}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.9
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness) # Получаем размер текста
                
                # Позиция текста: выше BBox, если есть место, иначе чуть ниже
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10 
                cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
            except ValueError as ve:
                print(f"Внимание: Ошибка преобразования данных в строке аннотации '{label_line.strip()}' из '{labels_path}': {ve}. Пропускаем.")
            except Exception as ex:
                print(f"Внимание: Неизвестная ошибка при обработке аннотации '{label_line.strip()}' из '{labels_path}': {ex}. Пропускаем.")

        # Сохранение изображения, если указан output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.basename(image_path).rsplit('.', 1)[0] + '_bbox.jpg' # Удаляем расширение и добавляем новое
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, img)
            print(f"Изображение с BBoxes сохранено в: {output_path}")

        # Отображение изображения в Jupyter
        if display_inline:
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8') # type: ignore
            display(HTML(f'<img src="data:image/jpeg;base64,{img_base64}" width="600">'))

    except Exception as e:
        print(f"Ошибка при отрисовке BBoxes для изображения '{image_path}': {e}")


def display_random_images_from_dir(
    directory: str,
    count: int = 5,
    title: str = "Примеры изображений:",
    img_width: int = 275
) -> None:
    """
    Отображает случайные изображения из указанной директории в виде HTML-строки в Jupyter Notebook.

    Args:
        directory (str): Полный или относительный путь к директории с изображениями.
        count (int): Количество случайных изображений для отображения.
        title (str): Заголовок, который будет выведен перед изображениями.
        img_width (int): Ширина каждого изображения в пикселях для HTML-отображения.
    """
    if not os.path.exists(directory):
        print(f"Ошибка: Директория не найдена: {directory}")
        return

    image_files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
    ]
    
    if len(image_files) == 0:
        print(f"Невозможно отобразить примеры, так как папка '{os.path.basename(directory)}' пуста.")
        return

    print(f"\n{title}")
    
    # Выбор до 'count' случайных изображений, если изображений меньше, выбираем все доступные
    example_images = random.sample(image_files, min(count, len(image_files)))
    
    html_content = ""
    for img_name in example_images:
        img_path = os.path.join(directory, img_name)
        # Используем base64 для надежного отображения в HTML, чтобы не зависеть от файловой системы Jupyter
        try:
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            # Добавлена минимальная стилизация для отступов
            html_content += f'<img src="data:image/jpeg;base64,{img_data}" style="width:{img_width}px; margin-right: 10px; display:inline-block;" title="{img_name}">'
        except FileNotFoundError:
            print(f"Внимание: Изображение '{img_name}' не найдено по пути '{img_path}'. Пропускаем.")
            continue
        except Exception as e:
            print(f"Ошибка при кодировании изображения '{img_name}': {e}. Пропускаем.")
            continue

    if html_content:
        display(HTML(html_content))
    else:
        print(f"Не удалось отобразить ни одно изображение из '{directory}'.")


def display_single_annotated_image_example(
    image_dir: str,
    annotations_dir: str,
    class_names: Dict[int, str],
    title: str = "Пример размеченного изображения:",
    display_annotation_content: bool = False
) -> None:
    """
    Выбирает случайное изображение из указанной директории, находит его аннотацию,
    отображает изображение с ограничивающими рамками и опционально выводит содержимое
    файла аннотации.

    Args:
        image_dir (str): Путь к директории с изображениями (например, 'resources/train_val_raw').
        annotations_dir (str): Путь к директории с файлами аннотаций YOLO (.txt).
        class_names (Dict[int, str]): Словарь, сопоставляющий ID класса с его именем.
        title (str): Заголовок для вывода перед отображением.
        display_annotation_content (bool): Если True, отображает содержимое .txt файла аннотации.
    """
    print(f"\n--- {title} ---")
    
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not image_files:
        print(f"Папка '{os.path.basename(image_dir)}' пуста, невозможно показать пример изображения с аннотацией.")
        return

    sample_image_name = random.choice(image_files)
    base_name = os.path.splitext(sample_image_name)[0]
    sample_image_path = os.path.join(image_dir, sample_image_name)
    sample_annotation_path = os.path.join(annotations_dir, base_name + '.txt')

    print(f"Выбранное изображение: '{sample_image_name}'")
    print(f"Предполагаемый путь к аннотации: '{sample_annotation_path}'")

    # Используем plot_bboxes_on_image для отображения
    plot_bboxes_on_image(
        image_path=sample_image_path,
        labels_path=sample_annotation_path,
        class_names=class_names,
        display_inline=True,
        output_dir=None # Не сохраняем в файл при отображении примера
    )

    # Отображаем содержимое .txt файла, если флаг установлен и файл существует
    if display_annotation_content:
        if os.path.exists(sample_annotation_path):
            print(f"\nСодержимое файла аннотации ('{base_name}.txt') - пример формата YOLO:")
            with open(sample_annotation_path, 'r') as f:
                annotation_content = f.read()
            display(Markdown(f"```txt\n{annotation_content}\n```"))
        else:
            print(f"Внимание: Файл аннотации '{sample_annotation_path}' не найден для демонстрации содержимого.")


def display_and_log_image_artifact(
    image_path: str,
    title: str,
    wandb_artifacts_dict: Optional[Dict[str, Any]] = None,
    wandb_key: Optional[str] = None,
    caption: Optional[str] = None,
    width: Optional[int] = None
) -> None:
    """
    Отображает изображение из файла в Jupyter Notebook и опционально логирует его как артефакт W&B.

    Args:
        image_path (str): Полный или относительный путь к файлу изображения.
        title (str): Заголовок для вывода в консоль перед отображением.
        wandb_artifacts_dict (Optional[Dict[str, Any]]): Словарь, в который будут добавлены W&B артефакты.
                                                       Если None, логирование в W&B не происходит.
        wandb_key (Optional[str]): Ключ для W&B артефакта (например, "test/PR_curve").
        caption (Optional[str]): Подпись для изображения в W&B.
        width (Optional[int]): Ширина отображаемого изображения в пикселях в Jupyter.
    """
    print(f"\n{title}:")
    try:
        if not os.path.exists(image_path):
            print(f"Ошибка: Файл изображения '{os.path.basename(image_path)}' не найден по пути: '{image_path}'. Невозможно отобразить или залогировать.")
            return

        display(Image(filename=image_path, width=width))
        
        # Логирование в W&B происходит, только если wandb установлен, словарь и ключ предоставлены
        if wandb_artifacts_dict is not None and wandb_key is not None and wandb is not None:
            wandb_artifacts_dict[wandb_key] = wandb.Image(image_path, caption=caption if caption else title)
        elif wandb_artifacts_dict is not None and wandb_key is None:
            print(f"Внимание: 'wandb_key' не указан для логирования артефакта '{image_path}' в W&B.")
        elif wandb is None and (wandb_artifacts_dict is not None or wandb_key is not None):
            print(f"Внимание: wandb не установлен или недоступен. Пропуск логирования артефакта '{image_path}'.")

    except Exception as e:
        print(f"Не удалось отобразить или залогировать артефакт '{os.path.basename(image_path)}' в W&B: {e}")


def display_and_log_multiple_image_artifacts(
    base_dir: str,
    image_filenames: List[str],
    prefix_title: str,
    wandb_artifacts_dict: Optional[Dict[str, Any]] = None,
    wandb_key_prefix: Optional[str] = None,
    widths: Optional[Dict[str, int]] = None
) -> None:
    """
    Отображает несколько изображений из указанной директории в Jupyter Notebook
    и опционально логирует их как артефакты W&B.

    Args:
        base_dir (str): Базовая директория, где находятся изображения.
        image_filenames (List[str]): Список имен файлов изображений для отображения.
        prefix_title (str): Префикс для заголовка, который будет выводиться перед каждым изображением.
        wandb_artifacts_dict (Optional[Dict[str, Any]]): Словарь, в который будут добавлены W&B артефакты.
        wandb_key_prefix (Optional[str]): Префикс для ключа W&B артефакта (например, "train/", "test/").
        widths (Optional[Dict[str, int]]): Словарь {filename: width} для установки индивидуальной ширины отображения.
                                           Если None, ширина не задается.
    """
    if widths is None:
        widths = {}

    if not os.path.exists(base_dir):
        print(f"Ошибка: Базовая директория '{base_dir}' не найдена. Невозможно отобразить или залогировать изображения.")
        return

    for filename in image_filenames:
        full_path = os.path.join(base_dir, filename)
        # Форматируем заголовок, заменяя подчеркивания и делая первую букву заглавной
        title = f"{prefix_title} {os.path.basename(filename).replace('.png', '').replace('.jpg', '').replace('_', ' ').capitalize()}"
        
        # Генерация ключа для W&B, если необходимо
        current_wandb_key: Optional[str] = None
        if wandb_artifacts_dict is not None and wandb_key_prefix is not None:
            clean_filename = os.path.splitext(filename)[0] # Извлекаем чистое имя файла без расширения
            current_wandb_key = f"{wandb_key_prefix}{clean_filename}"

        display_and_log_image_artifact(
            image_path=full_path,
            title=title,
            wandb_artifacts_dict=wandb_artifacts_dict,
            wandb_key=current_wandb_key,
            caption=title, # Используем заголовок как подпись для W&B
            width=widths.get(filename) # Получаем индивидуальную ширину, если она задана
        )


def create_side_by_side_demo_video(
    original_video_path: str,
    tracked_video_path: str,
    tracking_log_path: str,
    output_demo_path: str,
    tracked_video_fixed_size: int = 640,
    overlay_height: int = 120,
    original_video_width_ratio: float = 0.6
) -> None:
    """
    Создает комплексное демонстрационное видео, объединяющее исходное видео,
    отслеживаемое видео и подробный информационный оверлей с данными трекинга
    и командами дрона.

    Видео имеет макет: исходное видео (60% ширины) || отслеживаемое видео (40% ширины)
    с нижней секцией для оверлея.

    Args:
        original_video_path (str): Полный или относительный путь к исходному видеофайлу.
        tracked_video_path (str): Полный или относительный путь к видеофайлу с отслеживанием
                                  (с BBoxes и ID, созданный tracker.py).
        tracking_log_path (str): Полный или относительный путь к файлу логов трекинга (JSONL),
                                 созданному tracker.py.
        output_demo_path (str): Полный или относительный путь для сохранения объединенного демо-видео.
        tracked_video_fixed_size (int): Желаемый размер стороны квадратного отслеживаемого видео (например, 640 пикселей).
        overlay_height (int): Высота нижней области оверлея в пикселях.
        original_video_width_ratio (float): Доля ширины, которую занимает исходное видео
                                            в верхней секции (например, 0.6 для 60%).
    """
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_tracked = cv2.VideoCapture(tracked_video_path)

    # Проверка открытия видеофайлов
    if not cap_orig.isOpened():
        print(f"Ошибка: Не удалось открыть исходное видео по пути: '{original_video_path}'.")
        return
    if not cap_tracked.isOpened():
        print(f"Ошибка: Не удалось открыть отслеживаемое видео по пути: '{tracked_video_path}'.")
        cap_orig.release()
        return

    # Загружаем данные логов трекинга
    tracking_data: List[Dict[str, Any]] = [] # Явная аннотация типа
    try:
        if not os.path.exists(tracking_log_path):
            raise FileNotFoundError(f"Файл логов не найден: '{tracking_log_path}'.")
        with open(tracking_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                tracking_data.append(json.loads(line.strip()))
        print(f"Успешно загружены данные трекинга из: '{tracking_log_path}'. Всего записей: {len(tracking_data)}.")
    except FileNotFoundError as e:
        print(f"Ошибка: {e}. Оверлей может быть неполным или пустым.")
        # Заполнение заглушкой, если файл логов не найден
        num_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
        tracking_data = [{
            "frame": i + 1, "status": "НЕТ ЛОГОВ",
            "commands": {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"},
            "confidence": None, "bbox_size_ratio": None, "tracked_id": None
        } for i in range(num_frames)]
    except json.JSONDecodeError as e:
        print(f"Ошибка чтения файла логов '{tracking_log_path}': {e}. Проверьте формат файла. Оверлей будет неполным.")
        # Заполнение заглушкой при ошибке парсинга JSON
        num_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
        tracking_data = [{
            "frame": i + 1, "status": "ОШИБКА ЛОГОВ",
            "commands": {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"},
            "confidence": None, "bbox_size_ratio": None, "tracked_id": None
        } for i in range(num_frames)]
    except Exception as e:
        print(f"Неизвестная ошибка при загрузке логов '{tracking_log_path}': {e}. Оверлей будет неполным.")
        num_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
        tracking_data = [{
            "frame": i + 1, "status": "ОШИБКА ЛОГОВ",
            "commands": {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"},
            "confidence": None, "bbox_size_ratio": None, "tracked_id": None
        } for i in range(num_frames)]


    # Получаем свойства исходных видео
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Высота верхней секции будет определяться фиксированным размером отслеживаемого видео
    top_section_height = tracked_video_fixed_size

    # Общая ширина верхней секции, исходя из желаемого соотношения
    top_section_width = int(tracked_video_fixed_size / (1.0 - original_video_width_ratio) * original_video_width_ratio + tracked_video_fixed_size)

    # Рассчитываем ширину для исходного видео (60% от общей ширины верхней секции)
    new_orig_width = int(top_section_width * original_video_width_ratio)
    
    # Масштабируем исходное видео, сохраняя пропорции, и подгоняем его по высоте.
    # Это может потребовать добавления черных полос (padding) сверху/снизу,
    # если исходное видео не имеет идеального соотношения сторон для новой ширины.
    orig_aspect_ratio = orig_width / orig_height
    scaled_orig_height = int(new_orig_width / orig_aspect_ratio)

    # Общие размеры финального демо-видео
    output_width = top_section_width
    output_height = top_section_height + overlay_height

    # Подготовка шрифтов для отрисовки русского текста в оверлее (используем PIL)
    font_path = os.path.join(_PROJECT_ROOT, "resources", "arial.ttf")
    
    status_font_size = 24
    commands_font_size = 20
    # Отключено: fps_font_size = 20 # FPS логирование убрано из tracker.py

    try:
        status_font = ImageFont.truetype(font_path, status_font_size)
        commands_font = ImageFont.truetype(font_path, commands_font_size)
        # fps_font = ImageFont.truetype(font_path, fps_font_size) # Отключено
    except IOError:
        print(f"Внимание: Шрифт '{font_path}' не найден. Будет использоваться шрифт по умолчанию, что может привести к неправильному отображению кириллицы.")
        status_font = ImageFont.load_default()
        commands_font = ImageFont.load_default()
        # fps_font = ImageFont.load_default() # Отключено

    # Подготовка для записи выходного демо-видео
    output_dir = os.path.dirname(output_demo_path)
    os.makedirs(output_dir, exist_ok=True) # Добавлено exist_ok=True для надежности
    print(f"Создана/проверена выходная директория для демо-видео: '{output_dir}'")

    out: Optional[cv2.VideoWriter] = None # Явная аннотация типа
    # Порядок кодеков для повышения совместимости
    codec_options: List[Tuple[str, str]] = [
        ('mp4v', '.mp4'), # MPEG-4 Video Codec (хорошая совместимость)
        ('avc1', '.mp4'), # H.264 (распространенный, но может требовать дополнительных пакетов)
        ('XVID', '.avi'), # XVID MPEG-4 (старый, но универсальный)
        ('MJPG', '.avi')  # Motion JPEG (большой размер, но почти всегда поддерживается)
    ]

    final_output_path = output_demo_path

    for codec_fourcc_str, ext in codec_options:
        temp_output_path = output_demo_path.rsplit('.', 1)[0] + ext
        try:
            print(f"Попытка инициализации VideoWriter для демо-видео с кодеком '{codec_fourcc_str}' и расширением '{ext}'...")
            fourcc = cv2.VideoWriter_fourcc(*codec_fourcc_str) # type: ignore # Pylance может ругаться, но это рабочий синтаксис
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (output_width, output_height))
            if out.isOpened():
                final_output_path = temp_output_path
                print(f"Успешно инициализирован VideoWriter для демо-видео с кодеком '{codec_fourcc_str}'. Видео будет сохранено как: '{final_output_path}'.")
                break # Выходим из цикла, если удалось инициализировать
            else:
                print(f"Не удалось инициализировать VideoWriter для демо-видео с кодеком '{codec_fourcc_str}'.")
        except Exception as e:
            print(f"Ошибка при попытке инициализации VideoWriter для демо-видео с кодеком '{codec_fourcc_str}': {e}")
        out = None # Сбрасываем out, если попытка не удалась

    if out is None or not out.isOpened():
        print("Критическая ошибка: Не удалось создать VideoWriter для демо-видео ни с одним из предложенных кодеков. Проверьте установку кодеков, права доступа и работоспособность OpenCV.")
        cap_orig.release()
        cap_tracked.release()
        return # Выход из функции при критической ошибке
    
    output_demo_path = final_output_path # Обновляем путь на тот, который реально будет использован
    print(f"Создание демо-видео: '{output_demo_path}' с разрешением {output_width}x{output_height}.")

    frame_idx = 0
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_tracked, frame_tracked = cap_tracked.read()

        # Завершение цикла, если любой из видеопотоков закончился
        if not ret_orig or not ret_tracked:
            if not ret_orig: print("Исходное видео закончилось или ошибка чтения.")
            if not ret_tracked: print("Отслеживаемое видео закончилось или ошибка чтения.")
            break

        # 1. Подготовка верхней части: масштабирование и объединение видео
        # Масштабируем исходное видео до рассчитанной ширины
        current_orig_scaled = cv2.resize(frame_orig, (new_orig_width, scaled_orig_height))
        
        # Создаем пустой холст для исходного видео, чтобы добавить padding по высоте
        padded_orig_frame = np.zeros((top_section_height, new_orig_width, 3), dtype=np.uint8)
        # Центрируем масштабированное исходное видео по вертикали на холсте
        y_offset = (top_section_height - scaled_orig_height) // 2
        padded_orig_frame[y_offset : y_offset + scaled_orig_height, :] = current_orig_scaled

        # Изменяем размер отслеживаемого видео до фиксированного квадратного размера
        resized_tracked_frame = cv2.resize(frame_tracked, (tracked_video_fixed_size, tracked_video_fixed_size))
        
        # Объединяем два видеокадра по горизонтали
        top_section = np.hstack((padded_orig_frame, resized_tracked_frame))

        # 2. Подготовка нижней части: оверлей
        overlay_section = np.zeros((overlay_height, output_width, 3), dtype=np.uint8)
        
        # Получаем данные для текущего кадра из логов.
        # Если frame_idx превышает количество записей в логах, используем заглушку.
        current_log_data = tracking_data[frame_idx] if frame_idx < len(tracking_data) else {
            "frame": frame_idx + 1, "status": "Нет данных (лог)", # Изменено для ясности
            "commands": {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"},
            "confidence": None, "bbox_size_ratio": None, "tracked_id": None
            # "fps": None # Убрано, так как FPS логирование отключено
        }

        # --- ОТРИСОВКА ТЕКСТА С ИСПОЛЬЗОВАНИЕМ PILLOW ---
        # Преобразуем numpy массив OpenCV (BGR) в изображение PIL (RGB) для отрисовки текста
        pil_img = PILImage.fromarray(cv2.cvtColor(overlay_section, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        text_color_pil = (255, 255, 255) # Белый цвет для Pillow (RGB: Red, Green, Blue)

        # Строка 1: Статус трекинга и ID
        status_text = f"Статус: {current_log_data.get('status', 'N/A')}"
        if current_log_data.get('tracked_id') is not None:
             status_text += f" (ID: {current_log_data['tracked_id']})"
        draw.text((10, 5), status_text, font=status_font, fill=text_color_pil)

        # Строка 2: Уверенность и соотношение размера Bounding Box
        info_text = ""
        confidence = current_log_data.get('confidence')
        if confidence is not None:
            info_text += f"Уверенность: {confidence:.2f}"
        
        bbox_size_ratio = current_log_data.get('bbox_size_ratio')
        if bbox_size_ratio is not None:
            if info_text: info_text += " | "
            # Преобразуем строковое соотношение в число для логики статуса
            try:
                size_ratio_val = float(bbox_size_ratio.replace('%', '')) / 100.0
                size_status = ""
                if size_ratio_val < 0.9: size_status = "(Приблизься!)"
                elif size_ratio_val > 1.1: size_status = "(Отдалиться!)"
                else: size_status = "(ОК)"
                info_text += f"Размер: {bbox_size_ratio} {size_status}"
            except ValueError:
                info_text += f"Размер: {bbox_size_ratio} (ошибка парсинга)"

        draw.text((10, 35), info_text, font=commands_font, fill=text_color_pil)

        # Строка 3: Команды дрона (Горизонталь | Вертикаль | Дистанция)
        commands_dict = current_log_data.get('commands', {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"})
        commands_text = f"Команды: Гор: {commands_dict['horizontal']} | Верт: {commands_dict['vertical']} | Дист: {commands_dict['distance']}"
        draw.text((10, 65), commands_text, font=commands_font, fill=text_color_pil)

        # Отключено: Строка для FPS обработки (логирование FPS убрано из tracker.py)
        # if current_log_data.get('fps') is not None:
        #     fps_text = f"FPS обработки: {current_log_data['fps']:.1f}"
        #     text_width, text_height = draw.textsize(fps_text, font=fps_font) # type: ignore # Используется для устаревших методов, но функционально
        #     draw.text((output_width - text_width - 10, 5), fps_text, font=fps_font, fill=text_color_pil)

        # Преобразуем изображение PIL обратно в numpy массив OpenCV (RGB в BGR)
        overlay_section = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # -------------------------------------------------------------

        # Объединяем верхнюю и нижнюю секции для получения полного кадра демонстрационного видео
        combined_frame = np.vstack((top_section, overlay_section))
        
        # Записываем обработанный кадр в выходное видео
        out.write(combined_frame)
        frame_idx += 1
        if frame_idx % 100 == 0: # Периодический вывод прогресса
            print(f"Создано демо-кадров: {frame_idx}")

    # Освобождаем ресурсы VideoCapture и VideoWriter
    cap_orig.release()
    cap_tracked.release()
    if out is not None: # Проверяем, что VideoWriter был успешно инициализирован
        out.release()
    cv2.destroyAllWindows() # Закрываем все окна OpenCV (если они были открыты)
    print(f"Создание демонстрационного видео завершено. Результат сохранен в '{output_demo_path}'.")
