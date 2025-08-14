import os
import random
import shutil
from typing import List, Tuple, Optional # Импортируем типы для аннотаций

def split_train_val_dataset(
    images_dir: str, # Путь к папке с отобранными для обучения/валидации изображениями
    annotations_dir: str, # Путь к папке со всеми аннотациями
    base_dataset_dir: str, # Базовый путь для сохранения разделенного датасета (resources/dataset)
    val_split_ratio: float = 0.2, # Процент данных для валидации (по умолчанию 20%)
    random_seed: Optional[int] = 42 # Для воспроизводимости перемешивания 
) -> Tuple[int, int]:
    """
    Разделяет изображения и соответствующие аннотации на обучающую (train) и
    валидационную (val) выборки. Копирует файлы в структуру датасета YOLO.

    Args:
        images_dir (str): Путь к папке с изображениями, предназначенными для
                          разделения на train/val (e.g., 'resources/train_val_raw').
        annotations_dir (str): Путь к папке, содержащей все файлы аннотаций.
        base_dataset_dir (str): Базовый путь к целевой структуре датасета YOLO
                                (e.g., 'resources/dataset').
        val_split_ratio (float): Доля данных, которая будет отведена под валидационную выборку.
                                 Должно быть в диапазоне (0, 1).
        random_seed (Optional[int]): Опциональный seed для генератора случайных чисел.
                                     Использование seed обеспечивает воспроизводимость разделения.

    Returns:
        Tuple[int, int]: Кортеж, содержащий количество изображений в train и val наборах соответственно.
    """
    if not (0 < val_split_ratio < 1):
        print("Ошибка: val_split_ratio должен быть между 0 и 1.")
        return 0, 0

    if random_seed is not None:
        random.seed(random_seed)

    # Пути для сохранения разделенных данных
    train_images_dir = os.path.join(base_dataset_dir, 'images', 'train')
    val_images_dir = os.path.join(base_dataset_dir, 'images', 'val')

    train_labels_dir = os.path.join(base_dataset_dir, 'labels', 'train')
    val_labels_dir = os.path.join(base_dataset_dir, 'labels', 'val')

    # Создаем необходимые папки
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    print(f"Подготовка к разделению данных train/val из '{images_dir}'...")

    # Получаем список всех изображений из исходной папки
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    all_train_val_images = [
        f for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(image_extensions)
    ]
    random.shuffle(all_train_val_images) # Перемешиваем для случайного разделения

    num_val = int(len(all_train_val_images) * val_split_ratio)
    val_images = all_train_val_images[:num_val]
    train_images = all_train_val_images[num_val:]

    print(f"Всего изображений для train/val: {len(all_train_val_images)}")
    print(f"Количество изображений для обучения (Train): {len(train_images)} ({len(train_images)/len(all_train_val_images):.2%})")
    print(f"Количество изображений для валидации (Validation): {len(val_images)} ({len(val_images)/len(all_train_val_images):.2%})")

    # Вспомогательная функция для копирования файлов
    def _copy_files_to_split_dir(image_list: List[str], image_src_dir: str, image_dest_dir: str, label_src_dir: str, label_dest_dir: str) -> None:
        """Копирует изображения и их аннотации в целевые директории."""
        for img_name in image_list:
            # Копируем изображение
            src_image_path = os.path.join(image_src_dir, img_name)
            dest_image_path = os.path.join(image_dest_dir, img_name)
            try:
                shutil.copy(src_image_path, dest_image_path)
            except FileNotFoundError:
                print(f"Ошибка: Изображение '{src_image_path}' не найдено. Пропускаем.")
                continue
            except Exception as e:
                print(f"Ошибка при копировании изображения '{src_image_path}': {e}. Пропускаем.")
                continue

            # Копируем соответствующую аннотацию
            base_name = os.path.splitext(img_name)[0]
            label_name = base_name + '.txt'
            src_label_path = os.path.join(label_src_dir, label_name)
            dest_label_path = os.path.join(label_dest_dir, label_name)

            if os.path.exists(src_label_path):
                try:
                    shutil.copy(src_label_path, dest_label_path)
                except Exception as e:
                    print(f"Ошибка при копировании аннотации '{src_label_path}': {e}. Пропускаем.")
            else:
                print(f"Внимание: Файл аннотации '{label_name}' не найден для изображения '{img_name}'. Пропускаем копирование аннотации.")

    print("\nКопирование файлов для обучающей выборки...")
    _copy_files_to_split_dir(train_images, images_dir, train_images_dir, annotations_dir, train_labels_dir)

    print("Копирование файлов для валидационной выборки...")
    _copy_files_to_split_dir(val_images, images_dir, val_images_dir, annotations_dir, val_labels_dir)

    print("\nРазделение Train/Validation завершено.")
    return len(train_images), len(val_images)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    IMAGES_DIR_DEFAULT = os.path.join(project_root, 'resources', 'train_val_raw')
    ANNOTATIONS_DIR_DEFAULT = os.path.join(project_root, 'resources', 'annotations')
    BASE_DATASET_DIR_DEFAULT = os.path.join(project_root, 'resources', 'dataset')

    num_train, num_val = split_train_val_dataset(
        images_dir=IMAGES_DIR_DEFAULT,
        annotations_dir=ANNOTATIONS_DIR_DEFAULT,
        base_dataset_dir=BASE_DATASET_DIR_DEFAULT,
        val_split_ratio=0.2,
        random_seed=42
    )
    print(f"Общее количество изображений в Train: {num_train}")
    print(f"Общее количество изображений в Validation: {num_val}")