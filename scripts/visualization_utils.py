import cv2
import os
import random
import base64
import numpy as np
from typing import Tuple, Dict, Any 
from IPython.display import display, Image, HTML, Markdown
import json
from PIL import Image, ImageDraw, ImageFont

try:
    import wandb
except ImportError:
    wandb = None # W&B не установлен или недоступен


# Определяем корневую директорию проекта, исходя из расположения текущего скрипта
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))


# Функция для отображения обычных изображений (без bbox)
def display_image_inline(image_path: str, width: int = 600):
    """
    Отображает изображение в Jupyter Notebook.

    Args:
        image_path (str): Путь к файлу изображения.
        width (int): Ширина отображаемого изображения в пикселях.
    """
    try:
        display(Image(filename=image_path, width=width))
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути: {image_path}")
    except Exception as e:
        print(f"Ошибка при отображении изображения {image_path}: {e}")


# Функция для отображения изображения с аннотацией
def plot_bboxes_on_image(image_path: str, labels_path: str, class_names: dict, output_dir: str = None, display_inline: bool = True):
    """
    Рисует ограничивающие рамки на изображении на основе YOLO-аннотаций.

    Args:
        image_path (str): Путь к файлу изображения.
        labels_path (str): Путь к файлу аннотаций YOLO (.txt).
        class_names (dict): Словарь с соответствием ID класса и имени (например, {0: 'snowboarder'}).
        output_dir (str, optional): Директория для сохранения изображения с BBoxes. Если None, не сохраняется.
        display_inline (bool): Если True, отображает изображение в Jupyter Notebook.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Ошибка: Не удалось загрузить изображение по пути {image_path}")
            return

        h, w, _ = img.shape

        if not os.path.exists(labels_path):
            print(f"Внимание: Файл аннотаций не найден для {image_path} по пути {labels_path}. Отображаем изображение без BBoxes.")
            labels = []
        else:
            with open(labels_path, 'r') as f:
                labels = f.readlines()

        for label in labels:
            parts = list(map(float, label.strip().split()))
            class_id = int(parts[0])
            x_center, y_center, bbox_width, bbox_height = parts[1:]

            # Преобразование относительных координат в абсолютные
            x1 = int((x_center - bbox_width / 2) * w)
            y1 = int((y_center - bbox_height / 2) * h)
            x2 = int((x_center + bbox_width / 2) * w)
            y2 = int((y_center + bbox_height / 2) * h)

            # Отрисовка BBox
            color = (0, 255, 0) # Зеленый цвет
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Добавление метки класса
            class_name = class_names.get(class_id, f"Class {class_id}")
            text = f"{class_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10 # Позиция текста выше или ниже BBox
            cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # Сохранение изображения, если указан output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '_bbox.jpg'))
            cv2.imwrite(output_path, img)
            print(f"Изображение с BBoxes сохранено в: {output_path}")

        # Отображение изображения в Jupyter
        if display_inline:
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            display(HTML(f'<img src="data:image/jpeg;base64,{img_base64}" width="600">'))

    except Exception as e:
        print(f"Ошибка при отрисовке BBoxes для {image_path}: {e}")


def display_random_images_from_dir(directory: str, count: int = 5, title: str = "Примеры изображений:", img_width: int = 275):
    """
    Отображает случайные изображения из указанной директории в виде HTML-строки в Jupyter Notebook.

    Args:
        directory (str): Путь к директории с изображениями.
        count (int): Количество случайных изображений для отображения.
        title (str): Заголовок, который будет выведен перед изображениями.
        img_width (int): Ширина каждого изображения в пикселях для HTML-отображения.
    """
    if not os.path.exists(directory):
        print(f"Ошибка: Директория не найдена: {directory}")
        return

    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if len(image_files) == 0:
        print(f"Невозможно отобразить примеры, так как папка '{os.path.basename(directory)}' пуста.")
        return

    print(f"\n{title}")
    
    # Выбор до 'count' случайных изображений
    example_images = random.sample(image_files, min(count, len(image_files)))
    
    html_content = ""
    for img_name in example_images:
        img_path = os.path.join(directory, img_name)
        # Используем base64 для надежного отображения в HTML, чтобы не зависеть от файловой системы Jupyter
        try:
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            html_content += f'<img src="data:image/jpeg;base64,{img_data}" style="width:{img_width}px; margin-right: 10px; display:inline-block;" title="{img_name}">'
        except Exception as e:
            print(f"Ошибка при кодировании изображения {img_name}: {e}")
            continue

    if html_content:
        display(HTML(html_content))
    else:
        print(f"Не удалось отобразить ни одно изображение из {directory}.")


def display_single_annotated_image_example(
    image_dir: str,
    annotations_dir: str,
    class_names: dict,
    title: str = "Пример размеченного изображения:",
    display_annotation_content: bool = False
):
    """
    Выбирает случайное аннотированное изображение из указанной директории
    и отображает его с ограничивающими рамками.
    Может также отобразить содержимое файла аннотации.

    Args:
        image_dir (str): Путь к директории с изображениями (например, train_val_raw_dir или test_raw_dir).
        annotations_dir (str): Путь к директории с файлами аннотаций YOLO (.txt).
        class_names (dict): Словарь с соответствием ID класса и имени (например, {0: 'snowboarder'}).
        title (str): Заголовок для вывода перед изображением.
        display_annotation_content (bool): Если True, отображает содержимое .txt файла аннотации.
    """
    print(f"\n--- {title} ---")
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"Папка '{os.path.basename(image_dir)}' пуста, невозможно показать пример изображения с аннотацией.")
        return

    sample_image_name = random.choice(image_files)
    base_name = os.path.splitext(sample_image_name)[0]
    sample_image_path = os.path.join(image_dir, sample_image_name)
    sample_annotation_path = os.path.join(annotations_dir, base_name + '.txt')

    print(f"Изображение: {sample_image_name}")
    print(f"Путь к аннотации: {sample_annotation_path}")

    # Используем plot_bboxes_on_image для отображения
    plot_bboxes_on_image(
        image_path=sample_image_path,
        labels_path=sample_annotation_path,
        class_names=class_names,
        display_inline=True,
        output_dir=None
    )

    # Отображаем содержимое .txt файла, если флаг установлен и файл существует
    if display_annotation_content and os.path.exists(sample_annotation_path):
        print(f"\nСодержимое файла аннотации ({base_name}.txt) - пример формата YOLO:")
        with open(sample_annotation_path, 'r') as f:
            annotation_content = f.read()
        display(Markdown(f"```txt\n{annotation_content}\n```"))
    elif display_annotation_content and not os.path.exists(sample_annotation_path):
        print(f"Файл аннотации {sample_annotation_path} не найден для демонстрации содержимого.")


def display_and_log_image_artifact(
    image_path: str,
    title: str,
    wandb_artifacts_dict: dict = None, # Словарь для сбора артефактов W&B
    wandb_key: str = None,             # Ключ для W&B артефакта
    caption: str = None,               # Подпись для W&B артефакта
    width: int = None                  # Ширина для отображения в ноутбуке
):
    """
    Отображает изображение из файла в Jupyter Notebook и опционально логирует его как артефакт W&B.

    Args:
        image_path (str): Путь к файлу изображения.
        title (str): Заголовок для вывода в консоль перед отображением.
        wandb_artifacts_dict (dict, optional): Словарь, в который будут добавлены W&B артефакты.
                                                Если None, логирование в W&B не происходит через эту функцию.
        wandb_key (str, optional): Ключ для W&B артефакта (например, "test/PR_curve").
        caption (str, optional): Подпись для изображения в W&B.
        width (int, optional): Ширина отображаемого изображения в пикселях.
    """
    print(f"\n{title}:")
    try:
        if os.path.exists(image_path):
            display(Image(filename=image_path, width=width))
            if wandb_artifacts_dict is not None and wandb_key is not None and wandb is not None:
                wandb_artifacts_dict[wandb_key] = wandb.Image(image_path, caption=caption if caption else title)
        else:
            print(f"Файл '{os.path.basename(image_path)}' не найден по пути: {image_path}.")
    except Exception as e:
        print(f"Не удалось отобразить '{os.path.basename(image_path)}' или залогировать в W&B: {e}")


def display_and_log_multiple_image_artifacts(
    base_dir: str,
    image_filenames: list,
    prefix_title: str,
    wandb_artifacts_dict: dict = None,
    wandb_key_prefix: str = None, # Например, "train/" или "test/"
    widths: dict = None # Словарь {filename: width} для индивидуальной ширины
):
    """
    Отображает несколько изображений из указанной директории и опционально логирует их как артефакты W&B.

    Args:
        base_dir (str): Базовая директория, где находятся изображения.
        image_filenames (list): Список имен файлов изображений для отображения.
        prefix_title (str): Префикс для заголовка, который будет выводиться перед каждым изображением.
        wandb_artifacts_dict (dict, optional): Словарь, в который будут добавлены W&B артефакты.
        wandb_key_prefix (str, optional): Префикс для ключа W&B артефакта (например, "train/", "test/").
        widths (dict, optional): Словарь {filename: width} для установки индивидуальной ширины отображения.
                                  Если None, то ширина не задается.
    """
    if widths is None:
        widths = {}

    for filename in image_filenames:
        full_path = os.path.join(base_dir, filename)
        title = f"{prefix_title} {os.path.basename(filename).replace('.png', '').replace('.jpg', '').replace('_', ' ').capitalize()}"
        
        # Генерация ключа для W&B
        current_wandb_key = None
        if wandb_artifacts_dict is not None and wandb_key_prefix is not None:
            # Извлекаем чистое имя файла без расширения для ключа W&B
            clean_filename = os.path.splitext(filename)[0]
            current_wandb_key = f"{wandb_key_prefix}{clean_filename}"

        display_and_log_image_artifact(
            image_path=full_path,
            title=title,
            wandb_artifacts_dict=wandb_artifacts_dict,
            wandb_key=current_wandb_key,
            caption=title, # Используем заголовок как подпись
            width=widths.get(filename) # Получаем индивидуальную ширину
        )


def create_side_by_side_demo_video(
    original_video_path: str,
    tracked_video_path: str,
    tracking_log_path: str, # Путь к файлу логов с данными трекинга
    output_demo_path: str,
    tracked_video_fixed_size: int = 640, # Размер стороны квадратного отслеживаемого видео
    overlay_height: int = 120, # Высота нижней области оверлея
    original_video_width_ratio: float = 0.6 # Соотношение ширины исходного видео в верхней секции
) -> None:
    """
    Создает демонстрационное видео с исходным видео (60% ширины), отслеживаемым видео (40% ширины)
    и подробным текстовым оверлеем внизу.

    Args:
        original_video_path (str): Путь к исходному видеофайлу.
        tracked_video_path (str): Путь к видеофайлу с отслеживанием (с BBox и ID).
        tracking_log_path (str): Путь к файлу логов с данными трекинга (JSONL).
        output_demo_path (str): Путь для сохранения объединенного демо-видео.
        tracked_video_fixed_size (int): Размер стороны квадратного отслеживаемого видео (e.g., 640).
        overlay_height (int): Высота нижней области оверлея в пикселях.
        original_video_width_ratio (float): Доля ширины, которую занимает исходное видео (например, 0.6 для 60%).
    """
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_tracked = cv2.VideoCapture(tracked_video_path)

    if not cap_orig.isOpened():
        print(f"Ошибка: Не удалось открыть исходное видео {original_video_path}")
        return
    if not cap_tracked.isOpened():
        print(f"Ошибка: Не удалось открыть отслеживаемое видео {tracked_video_path}")
        cap_orig.release()
        return

    # Загружаем данные логов
    tracking_data = []
    try:
        with open(tracking_log_path, 'r') as f:
            for line in f:
                tracking_data.append(json.loads(line.strip()))
        print(f"Успешно загружены данные трекинга из: {tracking_log_path}")
    except FileNotFoundError:
        print(f"Ошибка: Файл логов не найден по пути: {tracking_log_path}. Оверлей может быть неполным.")
        tracking_data = [{"frame": i+1, "status": "НЕТ ЛОГОВ", "commands": {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"}, "confidence": None, "bbox_size_ratio": None, "tracked_id": None, "fps": None} for i in range(int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT)))]
    except json.JSONDecodeError as e:
        print(f"Ошибка чтения файла логов {tracking_log_path}: {e}. Проверьте формат файла.")
        tracking_data = [{"frame": i+1, "status": "ОШИБКА ЛОГОВ", "commands": {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"}, "confidence": None, "bbox_size_ratio": None, "tracked_id": None, "fps": None} for i in range(int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT)))]


    # Получаем свойства исходных видео
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Высота верхней секции будет определяться высотой отслеживаемого видео
    top_section_height = tracked_video_fixed_size

    # Общая ширина верхней секции
    top_section_width = int(tracked_video_fixed_size / (1.0 - original_video_width_ratio) * original_video_width_ratio + tracked_video_fixed_size) # Или просто int(tracked_video_fixed_size / (1 - original_video_width_ratio))

    # Рассчитываем ширину для исходного видео
    new_orig_width = int(top_section_width * original_video_width_ratio)
    
    # Масштабируем исходное видео, сохраняя пропорции, и подгоняем его по высоте к top_section_height
    # Чтобы избежать растягивания, мы масштабируем его до ширины new_orig_width
    # и центрируем по вертикали, добавляя черные полосы сверху/снизу
    orig_aspect_ratio = orig_width / orig_height
    scaled_orig_height = int(new_orig_width / orig_aspect_ratio)

    # Общие размеры финального демо-видео
    output_width = top_section_width
    output_height = top_section_height + overlay_height

    try:
        font_path = os.path.join(project_root, "resources", "arial.ttf")
        # Указываем размер шрифта (в пикселях)
        status_font_size = 24
        commands_font_size = 20
        fps_font_size = 20
        
        # Создаем объекты шрифтов
        status_font = ImageFont.truetype(font_path, status_font_size)
        commands_font = ImageFont.truetype(font_path, commands_font_size)
        fps_font = ImageFont.truetype(font_path, fps_font_size)
    except IOError:
        print(f"Внимание: Шрифт '{font_path}' не найден. Будет использоваться шрифт по умолчанию, что может привести к неправильному отображению кириллицы.")
        status_font = ImageFont.load_default()
        commands_font = ImageFont.load_default()
        fps_font = ImageFont.load_default()

    # Подготовка для записи выходного демо-видео
    output_dir = os.path.dirname(output_demo_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана выходная директория для демо-видео: {output_dir}")

    out = None
    codec_options = [
        ('mp4v', '.mp4'),
        ('avc1', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi')
    ]

    final_output_path = output_demo_path

    for codec_fourcc_str, ext in codec_options:
        temp_output_path = output_demo_path.rsplit('.', 1)[0] + ext
        try:
            print(f"Попытка инициализации VideoWriter для демо-видео с кодеком '{codec_fourcc_str}' и расширением '{ext}'...")
            fourcc = cv2.VideoWriter_fourcc(*codec_fourcc_str) # type: ignore
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (output_width, output_height))
            if out.isOpened():
                final_output_path = temp_output_path
                print(f"Успешно инициализирован VideoWriter для демо-видео с кодеком '{codec_fourcc_str}'. Видео будет сохранено как: {final_output_path}")
                break
            else:
                print(f"Не удалось инициализировать VideoWriter для демо-видео с кодеком '{codec_fourcc_str}'.")
        except Exception as e:
            print(f"Ошибка при попытке инициализации VideoWriter для демо-видео с кодеком '{codec_fourcc_str}': {e}")
        out = None

    if out is None or not out.isOpened():
        print("Критическая ошибка: Не удалось создать VideoWriter для демо-видео ни с одним из предложенных кодеков. Проверьте установку кодеков и права доступа.")
        cap_orig.release()
        cap_tracked.release()
        return
    
    output_demo_path = final_output_path
    print(f"Создание демо-видео: {output_demo_path} с разрешением {output_width}x{output_height}")

    frame_idx = 0
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_tracked, frame_tracked = cap_tracked.read()

        if not ret_orig or not ret_tracked:
            break

        # 1. Подготовка верхней части: масштабирование и объединение видео
        current_orig_scaled = cv2.resize(frame_orig, (new_orig_width, scaled_orig_height))
        padded_orig_frame = np.zeros((top_section_height, new_orig_width, 3), dtype=np.uint8)
        y_offset = (top_section_height - scaled_orig_height) // 2
        padded_orig_frame[y_offset : y_offset + scaled_orig_height, :] = current_orig_scaled

        resized_tracked_frame = cv2.resize(frame_tracked, (tracked_video_fixed_size, tracked_video_fixed_size))
        
        top_section = np.hstack((padded_orig_frame, resized_tracked_frame))

        # 2. Подготовка нижней части: оверлей
        overlay_section = np.zeros((overlay_height, output_width, 3), dtype=np.uint8)
        
        current_log_data = tracking_data[frame_idx] if frame_idx < len(tracking_data) else {
            "frame": frame_idx + 1, "status": "Нет данных", 
            "commands": {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"},
            "confidence": None, "bbox_size_ratio": None, "tracked_id": None, "fps": None
        }

        # --- ОТРИСОВКА ТЕКСТА С ИСПОЛЬЗОВАНИЕМ PILLOW ---
        pil_img = Image.fromarray(cv2.cvtColor(overlay_section, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        text_color_pil = (255, 255, 255) # Белый цвет для Pillow (RGB)

        # Строка 1: Статус трекинга и ID
        status_text = f"Статус: {current_log_data['status']}"
        if current_log_data.get('tracked_id') is not None:
             status_text += f" (ID: {current_log_data['tracked_id']})"
        draw.text((10, 5), status_text, font=status_font, fill=text_color_pil) # Позиция (x, y)

        # Строка 2: Уверенность и соотношение размера
        info_text = ""
        if current_log_data.get('confidence') is not None:
            info_text += f"Уверенность: {current_log_data['confidence']:.2f}"
        if current_log_data.get('bbox_size_ratio') is not None:
            if info_text: info_text += " | "
            size_ratio_val = float(current_log_data['bbox_size_ratio'].replace('%', '')) / 100.0
            size_status = ""
            if size_ratio_val < 0.9: size_status = "(Приблизься!)"
            elif size_ratio_val > 1.1: size_status = "(Отдалиться!)"
            else: size_status = "(ОК)"
            
            info_text += f"Размер: {current_log_data['bbox_size_ratio']} {size_status}"
        draw.text((10, 35), info_text, font=commands_font, fill=text_color_pil) # Позиция (x, y)

        # Строка 3: Команды дрона
        commands_dict = current_log_data.get('commands', {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"})
        commands_text = f"Команды: Гор: {commands_dict['horizontal']} | Верт: {commands_dict['vertical']} | Дист: {commands_dict['distance']}"
        draw.text((10, 65), commands_text, font=commands_font, fill=text_color_pil) # Позиция (x, y)

        # Строка 4: FPS обработки
        if current_log_data.get('fps') is not None:
            fps_text = f"FPS обработки: {current_log_data['fps']:.1f}"
            # Для позиционирования справа: вычисляем ширину текста и отнимаем от общей ширины
            text_width, text_height = draw.textsize(fps_text, font=fps_font) # type: ignore
            draw.text((output_width - text_width - 10, 5), fps_text, font=fps_font, fill=text_color_pil)

        # Преобразуем изображение PIL обратно в numpy массив для OpenCV
        overlay_section = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # -------------------------------------------------------------

        # Объединяем верхнюю и нижнюю секции
        combined_frame = np.vstack((top_section, overlay_section))
        
        out.write(combined_frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Создано демо-кадров: {frame_idx}")

    cap_orig.release()
    cap_tracked.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Создание демонстрационного видео завершено. Результат сохранен в {output_demo_path}")
