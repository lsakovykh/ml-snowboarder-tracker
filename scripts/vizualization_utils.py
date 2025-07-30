import cv2
import os
import random
import base64 
from IPython.display import display, Image, HTML, Markdown


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