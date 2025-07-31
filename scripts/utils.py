import os
import re
import yaml


def count_and_report_images(directory: str, description: str = "файлов", extensions=('.jpg', '.jpeg', '.png')):
    """
    Подсчитывает количество изображений в указанной директории и выводит отчет.

    Args:
        directory (str): Путь к директории.
        description (str): Описание подсчитываемых файлов (например, "извлеченных кадров").
        extensions (tuple): Кортеж расширений файлов, которые нужно учитывать.

    Returns:
        tuple: Кортеж, содержащий (список_файлов, количество_файлов).
               Возвращает ([], 0) если директория не существует или пуста.
    """
    if not os.path.exists(directory):
        print(f"Ошибка: Директория не найдена: '{directory}'.")
        return [], 0
    
    # Filter files based on extensions, case-insensitively
    files = [f for f in os.listdir(directory) if f.lower().endswith(extensions)]
    count = len(files)

    print(f"Общее количество {description} в '{directory}': {count}")
    return files, count


def verify_dataset_split(
    images_dir: str,
    labels_dir: str,
    data_split_name: str, # Например, "Train", "Validation", "Test"
    image_extensions=('.jpg', '.jpeg', '.png'),
    label_extension='.txt'
) -> bool:
    """
    Проверяет согласованность количества изображений и файлов аннотаций
    в указанных директориях и выводит отчет.

    Args:
        images_dir (str): Путь к директории с изображениями.
        labels_dir (str): Путь к директории с файлами аннотаций.
        data_split_name (str): Название подвыборки (например, "Обучающая", "Валидационная", "Тестовая").
        image_extensions (tuple): Кортеж расширений изображений для подсчета.
        label_extension (str): Расширение файла аннотации.

    Returns:
        bool: True, если количество изображений и аннотаций совпадает, False в противном случае.
    """
    
    # Подсчитываем изображения
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)] if os.path.exists(images_dir) else []
    images_count = len(image_files)

    # Подсчитываем файлы аннотаций
    label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith((label_extension,))] if os.path.exists(labels_dir) else []
    labels_count = len(label_files)
    
    print(f"{data_split_name} выборка (images): {images_count} изображений")
    print(f"{data_split_name} выборка (labels): {labels_count} аннотаций")

    if images_count == labels_count:
        print(f"\nКоличество изображений и аннотаций в выборке '{data_split_name}' совпадает. Разделение выполнено корректно.")
        return True
    else:
        print(f"\nВнимание: Количество изображений и аннотаций в выборке '{data_split_name}' НЕ совпадает. Проверьте соответствующие скрипты.")
        return False



def get_next_run_name(base_name: str, project_dir: str = '../runs/detect') -> str:
    """
    Определяет имя следующего запуска, автоматически инкрементируя номер версии.
    Например, для 'yolov8n_snowboarder' найдет 'yolov8n_snowboarder_v1', 'yolov8n_snowboarder_v2'
    и предложит 'yolov8n_snowboarder_v3'.

    Args:
        base_name (str): Базовое имя для запуска (например, 'yolov8n_snowboarder').
        project_dir (str): Директория, где хранятся запуски относительно корневой папки проекта.
                           По умолчанию '../runs/detect', так как скрипт запускается из ноутбука.

    Returns:
        str: Новое имя для запуска.
    """
    
    # Получаем текущую рабочую директорию (обычно папка notebooks/)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Переходим на уровень выше, чтобы попасть в корневую папку проекта
    project_root = os.path.join(current_script_dir, '..')

    # Строим полный путь к project_dir относительно корневой папки проекта
    full_project_dir = os.path.join(project_root, project_dir)

    if not os.path.exists(full_project_dir):
        os.makedirs(full_project_dir)
        return f"{base_name}_v1"

    pattern = re.compile(rf"^{re.escape(base_name)}_v(\d+)$")
    
    max_version = 0
    for folder_name in os.listdir(full_project_dir):
        match = pattern.match(folder_name)
        if match:
            try:
                version = int(match.group(1))
                if version > max_version:
                    max_version = version
            except ValueError:
                pass
    
    next_version = max_version + 1
    return f"{base_name}_v{next_version}"


def check_yolo_dataset_paths(yaml_path: str) -> bool:
    """
    Читает файл dataset.yaml, проверяет доступность всех указанных в нем путей
    для изображений и аннотаций, и выводит отчет.

    Args:
        yaml_path (str): Путь к файлу dataset.yaml.

    Returns:
        bool: True, если все пути доступны; False в противном случае.
    """
    all_paths_ok = True
    print(f"Содержимое файла конфигурации '{yaml_path}':")
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            yaml_content = yaml.safe_load(file)
            print(yaml.dump(yaml_content, indent=2))
        
        # Получаем базовый путь из YAML-файла
        base_path = yaml_content.get('path')
        if not base_path:
            print("ОШИБКА: Поле 'path' отсутствует в dataset.yaml.")
            return False

        # Убедимся, что base_path является абсолютным или правильным относительным
        abs_base_path = os.path.abspath(os.path.join(os.path.dirname(yaml_path), base_path))
        print(f"\nАбсолютный базовый путь датасета: {abs_base_path}")

        print("\nПроверка доступности путей изображений:")
        image_splits = {'train': 'train', 'val': 'val', 'test': 'test'}
        for key, name in image_splits.items():
            relative_path = yaml_content.get(key, '')
            full_path = os.path.join(abs_base_path, relative_path)
            
            status = 'Доступен' if os.path.exists(full_path) else 'ОШИБКА: Недоступен!'
            print(f"{name.capitalize()} images: {full_path} - {status}")
            if not os.path.exists(full_path):
                all_paths_ok = False
        
        # Проверка путей аннотаций (исправленная логика)
        labels_base_path = os.path.join(abs_base_path, 'labels')
        print(f"\nБазовый путь аннотаций: {labels_base_path}")
        print("Проверка доступности путей аннотаций:")

        label_splits = {'train': 'train', 'val': 'val', 'test': 'test'} # Используем те же ключи для label-поддиректорий
        for key, name in label_splits.items():
            full_path = os.path.join(labels_base_path, name) # Пути к labels всегда 'train', 'val', 'test'
            
            status = 'Доступен' if os.path.exists(full_path) else 'ОШИБКА: Недоступен!'
            print(f"{name.capitalize()} labels: {full_path} - {status}")
            if not os.path.exists(full_path):
                all_paths_ok = False
        
    except FileNotFoundError:
        print(f"Ошибка: Файл '{yaml_path}' не найден. Убедитесь, что он существует по указанному пути.")
        all_paths_ok = False
    except Exception as e:
        print(f"Ошибка при чтении или обработке YAML файла: {e}")
        all_paths_ok = False
        
    return all_paths_ok