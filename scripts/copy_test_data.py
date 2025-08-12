import os
import shutil
from typing import Tuple # Импортируем Tuple для аннотации типов

def copy_test_data(
    source_test_images_dir: str,
    all_annotations_dir: str,
    dest_dataset_base_dir: str
) -> Tuple[int, int]:
    """
    Копирует изображения и соответствующие файлы аннотаций из сырых папок
    в целевые папки тестового набора датасета YOLO.

    Args:
        source_test_images_dir (str): Путь к папке с сырыми изображениями тестового набора.
        all_annotations_dir (str): Путь к папке, содержащей все файлы аннотаций.
        dest_dataset_base_dir (str): Базовый путь к целевой структуре датасета YOLO (например, 'resources/dataset').

    Returns:
        Tuple[int, int]: Кортеж, содержащий количество скопированных изображений и количество скопированных аннотаций.
    """
    # Определяем целевые папки в вашей структуре dataset
    dest_test_images_dir = os.path.join(dest_dataset_base_dir, 'images', 'test')
    dest_test_labels_dir = os.path.join(dest_dataset_base_dir, 'labels', 'test')

    # Убедимся, что целевые папки существуют
    os.makedirs(dest_test_images_dir, exist_ok=True)
    os.makedirs(dest_test_labels_dir, exist_ok=True)

    print(f"Копирование тестовых данных из {source_test_images_dir} и {all_annotations_dir} в {dest_dataset_base_dir}...")

    # Получаем список имен файлов изображений из сырой тестовой папки
    # Учитываем только поддерживаемые форматы изображений
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    test_image_names = [
        f for f in os.listdir(source_test_images_dir)
        if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(source_test_images_dir, f))
    ]

    copied_images_count = 0
    copied_labels_count = 0

    for img_name in test_image_names:
        base_name = os.path.splitext(img_name)[0] # Имя файла без расширения

        # Копируем изображение
        src_image_path = os.path.join(source_test_images_dir, img_name)
        dest_image_path = os.path.join(dest_test_images_dir, img_name)
        
        try:
            shutil.copy(src_image_path, dest_image_path)
            copied_images_count += 1
        except FileNotFoundError:
            print(f"Ошибка: Исходное изображение '{src_image_path}' не найдено. Пропускаем.")
            continue
        except Exception as e:
            print(f"Ошибка при копировании изображения '{src_image_path}': {e}. Пропускаем.")
            continue

        # Копируем соответствующую аннотацию
        label_name = base_name + '.txt'
        src_label_path = os.path.join(all_annotations_dir, label_name)
        dest_label_path = os.path.join(dest_test_labels_dir, label_name)

        if os.path.exists(src_label_path):
            try:
                shutil.copy(src_label_path, dest_label_path)
                copied_labels_count += 1
            except Exception as e:
                print(f"Ошибка при копировании аннотации '{src_label_path}': {e}. Пропускаем.")
        else:
            print(f"Внимание: Файл аннотации '{label_name}' не найден для изображения '{img_name}' в '{all_annotations_dir}'. Пропускаем копирование аннотации.")

    print(f"\nЗавершено копирование тестовых данных.")
    print(f"Скопировано изображений: {copied_images_count}")
    print(f"Скопировано файлов аннотаций: {copied_labels_count}")
    
    return copied_images_count, copied_labels_count

if __name__ == "__main__":
    # Эти параметры могут быть настроены при запуске скрипта напрямую
    # Убедитесь, что пути указаны относительно корневой директории, если скрипт запускается из scripts/
    # или используйте os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources', ...))
    
    # Для использования из ноутбука, где рабочая директория notebooks/, пути должны быть '../resources/...'
    # При прямом запуске из scripts/, пути должны быть 'resources/...'
    # Чтобы сделать скрипт независимым от рабочей директории при прямом запуске:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    SOURCE_TEST_IMAGES_DIR_DEFAULT = os.path.join(project_root, 'resources', 'test_raw')
    ALL_ANNOTATIONS_DIR_DEFAULT = os.path.join(project_root, 'resources', 'annotations')
    DEST_DATASET_BASE_DIR_DEFAULT = os.path.join(project_root, 'resources', 'dataset')

    copied_imgs, copied_labels = copy_test_data(
        source_test_images_dir=SOURCE_TEST_IMAGES_DIR_DEFAULT,
        all_annotations_dir=ALL_ANNOTATIONS_DIR_DEFAULT,
        dest_dataset_base_dir=DEST_DATASET_BASE_DIR_DEFAULT
    )
    print(f"Общее количество скопированных изображений: {copied_imgs}")
    print(f"Общее количество скопированных аннотаций: {copied_labels}")