import os
import re
import yaml
from typing import List, Tuple, Optional

# Определяем корневую директорию проекта, исходя из расположения текущего скрипта.
# Это позволяет функциям корректно работать с путями относительно корня проекта,
# независимо от того, откуда запущен Jupyter Notebook или скрипт.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))

def count_and_report_images(
    directory: str,
    description: str = "файлов",
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')
) -> Tuple[List[str], int]:
    """
    Подсчитывает количество файлов с указанными расширениями в директории
    и выводит отчет.

    Args:
        directory (str): Путь к директории для сканирования.
        description (str): Описание подсчитываемых файлов (например, "извлеченных кадров").
        extensions (Tuple[str, ...]): Кортеж расширений файлов, которые нужно учитывать
                                      (например, ('.jpg', '.jpeg', '.png')).

    Returns:
        Tuple[List[str], int]: Кортеж, содержащий:
                               - Список имен найденных файлов (без пути).
                               - Общее количество найденных файлов.
                               Возвращает ([], 0) если директория не существует или пуста.
    """
    if not os.path.exists(directory):
        print(f"Ошибка: Директория не найдена: '{directory}'.")
        return [], 0
    
    # Фильтруем файлы по расширениям (без учета регистра) и убеждаемся, что это именно файлы
    files = [
        f for f in os.listdir(directory) 
        if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(extensions)
    ]
    count = len(files)

    print(f"Общее количество {description} в '{directory}': {count}")
    return files, count


def verify_dataset_split(
    images_dir: str,
    labels_dir: str,
    data_split_name: str, # Например, "Train", "Validation", "Test"
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
    label_extension: str = '.txt'
) -> bool:
    """
    Проверяет согласованность количества изображений и файлов аннотаций
    в указанных директориях и выводит отчет.

    Args:
        images_dir (str): Путь к директории с изображениями.
        labels_dir (str): Путь к директории с файлами аннотаций.
        data_split_name (str): Название подвыборки (например, "Обучающая", "Валидационная", "Тестовая").
        image_extensions (Tuple[str, ...]): Кортеж расширений изображений для подсчета.
        label_extension (str): Расширение файла аннотации (например, '.txt').

    Returns:
        bool: True, если количество изображений и аннотаций совпадает и обе директории существуют;
              False в противном случае.
    """
    # Проверка существования директорий перед попыткой чтения их содержимого
    images_dir_exists = os.path.exists(images_dir)
    labels_dir_exists = os.path.exists(labels_dir)

    if not images_dir_exists:
        print(f"Ошибка: Директория изображений не найдена: '{images_dir}'.")
    if not labels_dir_exists:
        print(f"Ошибка: Директория аннотаций не найдена: '{labels_dir}'.")

    # Подсчитываем изображения, только если директория существует
    image_files: List[str] = [
        f for f in os.listdir(images_dir) 
        if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(image_extensions)
    ] if images_dir_exists else []
    images_count = len(image_files)

    # Подсчитываем файлы аннотаций, только если директория существует
    label_files: List[str] = [
        f for f in os.listdir(labels_dir) 
        if os.path.isfile(os.path.join(labels_dir, f)) and f.lower().endswith((label_extension,))
    ] if labels_dir_exists else []
    labels_count = len(label_files)
    
    print(f"{data_split_name} выборка (изображений): {images_count}")
    print(f"{data_split_name} выборка (аннотаций): {labels_count}")

    # Итоговая проверка согласованности
    if not images_dir_exists or not labels_dir_exists:
        print(f"Внимание: Не удалось проверить согласованность для '{data_split_name}' из-за отсутствия одной или обеих директорий.")
        return False
    elif images_count == labels_count:
        print(f"\nКоличество изображений и аннотаций в выборке '{data_split_name}' совпадает. Разделение выполнено корректно.")
        return True
    else:
        print(f"\nВнимание: Количество изображений и аннотаций в выборке '{data_split_name}' НЕ совпадает. Проверьте соответствующие скрипты.")
        return False


def get_next_run_name(base_name: str, runs_relative_path: str = 'runs/detect') -> str:
    """
    Определяет имя следующего запуска, автоматически инкрементируя номер версии.
    Например, для 'yolo11n_snowboarder_detection' найдет 'yolo11n_snowboarder_detection_v1',
    'yolo11n_snowboarder_detection_v2' и предложит 'yolo11n_snowboarder_detection_v3'.

    Args:
        base_name (str): Базовое имя для запуска (например, 'yolo11n_snowboarder_detection').
                         Это префикс, который будет использоваться для поиска существующих запусков.
        runs_relative_path (str): Путь к директории, где хранятся запуски,
                                  относительно корневой папки проекта (например, 'runs/detect').

    Returns:
        str: Новое уникальное имя для запуска.
    """
    # Строим полный путь к директории, где ищутся запуски, относительно корня проекта.
    full_runs_dir = os.path.join(_PROJECT_ROOT, runs_relative_path)

    # Если директория запусков не существует, создаем её (exist_ok=True предотвращает ошибку, если она уже есть)
    os.makedirs(full_runs_dir, exist_ok=True)
    
    # Шаблон регулярного выражения для поиска папок вида 'base_name_vX'
    pattern = re.compile(rf"^{re.escape(base_name)}_v(\d+)$")
    
    max_version = 0
    # Перебираем все элементы в директории запусков
    for folder_name in os.listdir(full_runs_dir):
        # Проверяем, является ли элемент директорией (чтобы не обрабатывать файлы)
        item_full_path = os.path.join(full_runs_dir, folder_name)
        if os.path.isdir(item_full_path):
            match = pattern.match(folder_name)
            if match:
                try:
                    # Извлекаем номер версии и обновляем max_version
                    version = int(match.group(1))
                    if version > max_version:
                        max_version = version
                except ValueError:
                    # Игнорируем папки, если числовая часть не является корректным целым числом (маловероятно, но для робастности)
                    pass
    
    # Следующая версия будет на 1 больше максимальной найденной
    next_version = max_version + 1
    return f"{base_name}_v{next_version}"


def check_yolo_dataset_paths(yaml_path: str) -> bool:
    """
    Читает файл dataset.yaml, проверяет доступность всех указанных в нем путей
    для изображений и аннотаций, и выводит подробный отчет.

    Args:
        yaml_path (str): Путь к файлу dataset.yaml.

    Returns:
        bool: True, если файл yaml существует и все указанные в нем пути доступны;
              False в противном случае.
    """
    all_paths_ok = True
    print(f"Содержимое файла конфигурации '{yaml_path}':")
    try:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Файл '{yaml_path}' не найден.")

        with open(yaml_path, 'r', encoding='utf-8') as file:
            yaml_content = yaml.safe_load(file)
            print(yaml.dump(yaml_content, indent=2))
        
        if not isinstance(yaml_content, dict): # Дополнительная проверка типа содержимого YAML
            print(f"ОШИБКА: Содержимое файла '{yaml_path}' не является корректным словарем YAML.")
            return False

        # Получаем базовый путь из YAML-файла
        base_path_relative: Optional[str] = yaml_content.get('path')
        if not base_path_relative:
            print("ОШИБКА: Поле 'path' отсутствует или пусто в dataset.yaml. Убедитесь, что оно указано.")
            return False

        # Абсолютный базовый путь датасета (относительно директории, где лежит dataset.yaml)
        # Это важно, так как поле 'path' в dataset.yaml может быть относительным к самому YAML файлу.
        abs_base_path = os.path.abspath(os.path.join(os.path.dirname(yaml_path), base_path_relative))
        print(f"\nАбсолютный базовый путь датасета: {abs_base_path}")

        print("\nПроверка доступности путей изображений:")
        # Ключи, которые обычно используются для путей к изображениям в dataset.yaml
        image_splits_keys = ['train', 'val', 'test'] 
        for key in image_splits_keys:
            relative_path: Optional[str] = yaml_content.get(key)
            if not relative_path:
                print(f"Внимание: Поле '{key}' для изображений отсутствует или пусто в dataset.yaml. Этот раздел данных может быть не использован.")
                continue # Продолжаем, так как отсутствие пути не всегда является критической ошибкой

            full_path = os.path.join(abs_base_path, relative_path)
            
            status = 'Доступен' if os.path.exists(full_path) else 'ОШИБКА: Недоступен!'
            print(f"{key.capitalize()} images: {full_path} - {status}")
            if not os.path.exists(full_path): # Дополнительная проверка на существование
                all_paths_ok = False
        
        # Проверка путей аннотаций
        print("\nПроверка доступности путей аннотаций:")
        # В YOLO аннотации обычно находятся в подпапках 'labels/train', 'labels/val', 'labels/test'
        # относительно базового пути датасета.
        # Если в dataset.yaml указано поле 'labels' (например, 'labels: ../labels'), используем его.
        # В противном случае, по умолчанию ожидаем 'labels' внутри базовой папки датасета.
        labels_base_path_in_yaml: Optional[str] = yaml_content.get('labels')
        
        if labels_base_path_in_yaml:
            # Если поле 'labels' указано, строим путь относительно директории YAML-файла
            full_labels_base_path = os.path.abspath(os.path.join(os.path.dirname(yaml_path), labels_base_path_in_yaml))
        else:
            # Если поле 'labels' отсутствует, предполагаем стандартную структуру: 'labels' внутри abs_base_path
            full_labels_base_path = os.path.join(abs_base_path, 'labels')

        print(f"Предполагаемый базовый путь аннотаций: {full_labels_base_path}")

        label_splits_keys = ['train', 'val', 'test'] 
        for key in label_splits_keys:
            # Полный путь к подпапке аннотаций
            full_path = os.path.join(full_labels_base_path, key) 
            
            status = 'Доступен' if os.path.exists(full_path) else 'ОШИБКА: Недоступен!'
            print(f"{key.capitalize()} labels: {full_path} - {status}")
            if not os.path.exists(full_path): # Дополнительная проверка на существование
                all_paths_ok = False
        
    except FileNotFoundError as e:
        print(f"ОШИБКА: {e}. Убедитесь, что файл '{yaml_path}' существует по указанному пути.")
        all_paths_ok = False
    except yaml.YAMLError as e:
        print(f"ОШИБКА парсинга YAML файла '{yaml_path}': {e}. Проверьте синтаксис YAML.")
        all_paths_ok = False
    except Exception as e:
        print(f"НЕИЗВЕСТНАЯ ОШИБКА при чтении или обработке YAML файла '{yaml_path}': {e}")
        all_paths_ok = False
            
    return all_paths_ok