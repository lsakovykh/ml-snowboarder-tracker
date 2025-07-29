import os

# Пути к папкам
ALL_IMAGES_DIR = 'resources/all_frames' # Это папка, где лежат все изображениz, извлеченные из видео.
TRAIN_VAL_IMAGES_DIR = 'resources/train_val_raw' # Папка с 207 отобранными для обучения изображениями.

# Получаем список имен файлов (только имена, без пути)
all_image_names = set(f for f in os.listdir(ALL_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
train_val_image_names = set(f for f in os.listdir(TRAIN_VAL_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

# Находим имена файлов, которые есть в all_images_names, но НЕТ в selected_image_names
unselected_image_names = sorted(list(all_image_names - train_val_image_names))

print(f"Всего извлечено изображений (All): {len(all_image_names)}")
print(f"Уже размечено изображений (Selected): {len(train_val_image_names)}")
print(f"Доступно для разметки на тестовый набор (Unselected): {len(unselected_image_names)}")

# Выведем первые несколько имен файлов, которые можно разметить
print("\nНеразмеченные изображения, доступные для тестового набора (первые 10):")
for i, name in enumerate(unselected_image_names):
    print(f"- {name}")

# Можете сохранить весь список в файл, чтобы удобно было выбирать
with open('unselected_images_for_test.txt', 'w') as f:
    for name in unselected_image_names:
        f.write(name + '\n')

print(f"\nПолный список неразмеченных изображений сохранен в 'unselected_images_for_test.txt'")