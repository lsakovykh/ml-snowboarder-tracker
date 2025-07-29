import os
import shutil

# --- Настройка путей (относительно папки scripts) ---
# Папка с сырыми изображениями тестового набора
SOURCE_TEST_IMAGES_DIR = 'resources/test_raw'
# Папка со всеми аннотациями
ALL_ANNOTATIONS_DIR = 'resources/annotations'

# Целевые папки в вашей структуре dataset
DEST_DATASET_BASE_DIR = 'resources/dataset'
DEST_TEST_IMAGES_DIR = os.path.join(DEST_DATASET_BASE_DIR, 'images', 'test')
DEST_TEST_LABELS_DIR = os.path.join(DEST_DATASET_BASE_DIR, 'labels', 'test')

# --- Убедимся, что целевые папки существуют ---
os.makedirs(DEST_TEST_IMAGES_DIR, exist_ok=True)
os.makedirs(DEST_TEST_LABELS_DIR, exist_ok=True)

print(f"Копирование тестовых данных из {SOURCE_TEST_IMAGES_DIR} и {ALL_ANNOTATIONS_DIR} в {DEST_DATASET_BASE_DIR}...")

# Получаем список имен файлов изображений из сырой тестовой папки
test_image_names = [f for f in os.listdir(SOURCE_TEST_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

copied_images_count = 0
copied_labels_count = 0

for img_name in test_image_names:
    base_name = os.path.splitext(img_name)[0] # Имя файла без расширения

    # Копируем изображение
    src_image_path = os.path.join(SOURCE_TEST_IMAGES_DIR, img_name)
    dest_image_path = os.path.join(DEST_TEST_IMAGES_DIR, img_name)
    shutil.copy(src_image_path, dest_image_path)
    copied_images_count += 1

    # Копируем соответствующую аннотацию
    label_name = base_name + '.txt'
    src_label_path = os.path.join(ALL_ANNOTATIONS_DIR, label_name)
    dest_label_path = os.path.join(DEST_TEST_LABELS_DIR, label_name)

    if os.path.exists(src_label_path):
        shutil.copy(src_label_path, dest_label_path)
        copied_labels_count += 1
    else:
        print(f"Warning: Annotation file {label_name} not found for image {img_name} in {ALL_ANNOTATIONS_DIR}. Skipping label copy for this image.")

print(f"\nЗавершено копирование тестовых данных.")
print(f"Скопировано изображений: {copied_images_count}")
print(f"Скопировано файлов аннотаций: {copied_labels_count}")