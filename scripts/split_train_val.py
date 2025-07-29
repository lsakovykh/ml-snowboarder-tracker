import os
import random
import shutil

# Пути к исходным данным 
IMAGES_DIR = 'resources/train_val_raw' # Отобранные для обучения кадры
ANNOTATIONS_DIR = 'resources/annotations' # Все аннотации

print(IMAGES_DIR)

# Пути для сохранения разделенных данных (теперь относительно папки scripts)
BASE_DATASET_DIR = 'resources/dataset'

TRAIN_IMAGES_DIR = os.path.join(BASE_DATASET_DIR, 'images', 'train')
VAL_IMAGES_DIR = os.path.join(BASE_DATASET_DIR, 'images', 'val')

TRAIN_LABELS_DIR = os.path.join(BASE_DATASET_DIR, 'labels', 'train')
VAL_LABELS_DIR = os.path.join(BASE_DATASET_DIR, 'labels', 'val')

# Процент данных для валидации
VAL_SPLIT_RATIO = 0.2

# Создаем необходимые папки
os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(VAL_IMAGES_DIR, exist_ok=True)

os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
os.makedirs(VAL_LABELS_DIR, exist_ok=True)

# Получаем список всех изображений из train_val_raw
all_train_val_images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_train_val_images) # Перемешиваем для случайного разделения

num_val = int(len(all_train_val_images) * VAL_SPLIT_RATIO)

val_images = all_train_val_images[:num_val]
train_images = all_train_val_images[num_val:]

print(f"Total train/val raw images: {len(all_train_val_images)}")
print(f"Train images: {len(train_images)} ({len(train_images)/len(all_train_val_images):.2%})")
print(f"Validation images: {len(val_images)} ({len(val_images)/len(all_train_val_images):.2%})")

# Копируем файлы
def copy_files(image_list, image_dest_dir, label_dest_dir):
    for img_name in image_list:
        # Копируем изображение
        shutil.copy(os.path.join(IMAGES_DIR, img_name), os.path.join(image_dest_dir, img_name))

        # Копируем соответствующую аннотацию из общей папки annotations
        base_name = os.path.splitext(img_name)[0]
        label_name = base_name + '.txt'
        src_label_path = os.path.join(ANNOTATIONS_DIR, label_name)
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, os.path.join(label_dest_dir, label_name))
        else:
            print(f"Warning: Annotation file {label_name} not found for image {img_name} in {ANNOTATIONS_DIR}. Skipping.")

print("Copying training files...")
copy_files(train_images, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)

print("Copying validation files...")
copy_files(val_images, VAL_IMAGES_DIR, VAL_LABELS_DIR)

print("Train/Validation split complete.")