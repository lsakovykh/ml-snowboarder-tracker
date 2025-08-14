import os
from typing import List, Set, Tuple # Импортируем типы для аннотаций

def find_unselected_frames(
    all_images_dir: str,
    train_val_images_dir: str,
    output_filename: str = 'unselected_images_for_test.txt'
) -> Tuple[List[str], str]:
    """
    Находит изображения, которые были извлечены из исходного видео, но не были
    использованы в обучающей или валидационной выборках. Сохраняет список
    этих изображений в текстовый файл.

    Args:
        all_images_dir (str): Путь к папке со всеми извлеченными кадрами из видео.
        train_val_images_dir (str): Путь к папке с изображениями,
                                    отобранными для обучения и валидации.
        output_filename (str): Имя файла, в который будет сохранен список невыбранных изображений.

    Returns:
        Tuple[List[str], str]: Кортеж, содержащий:
                               - Список имен невыбранных изображений (List[str]).
                               - Абсолютный путь к файлу, в который был сохранен список (str).
    """
    # Определяем поддерживаемые расширения изображений
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    # Получаем список имен файлов (только имена, без пути) из обеих папок
    # Проверяем, что это файлы, а не поддиректории
    all_image_names = set(
        f for f in os.listdir(all_images_dir)
        if os.path.isfile(os.path.join(all_images_dir, f)) and f.lower().endswith(image_extensions)
    )
    train_val_image_names = set(
        f for f in os.listdir(train_val_images_dir)
        if os.path.isfile(os.path.join(train_val_images_dir, f)) and f.lower().endswith(image_extensions)
    )

    # Находим имена файлов, которые есть в all_images_names, но НЕТ в train_val_image_names
    unselected_image_names = sorted(list(all_image_names - train_val_image_names))

    print(f"Всего извлечено изображений (All): {len(all_image_names)}")
    print(f"Уже отобрано для обучения/валидации (Train/Val): {len(train_val_image_names)}")
    print(f"Доступно для разметки на тестовый набор (Unselected): {len(unselected_image_names)}")

    # Выведем первые несколько имен файлов для предварительного просмотра
    print("\nНеразмеченные изображения, доступные для тестового набора (первые 10):")
    for i, name in enumerate(unselected_image_names[:10]): # Показываем только первые 10
        print(f"- {name}")
    if len(unselected_image_names) > 10:
        print(f"- ... (всего {len(unselected_image_names)} невыбранных изображений)")

    # Сохраняем весь список в файл
    output_filepath = os.path.abspath(output_filename) # Получаем абсолютный путь к выходному файлу
    with open(output_filepath, 'w') as f:
        for name in unselected_image_names:
            f.write(name + '\n')

    print(f"\nПолный список неразмеченных изображений сохранен в '{output_filepath}'")
    
    return unselected_image_names, output_filepath

if __name__ == "__main__":
    # Эти параметры могут быть настроены при запуске скрипта напрямую
    # Убедитесь, что пути указаны относительно корневой директории проекта
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    ALL_IMAGES_DIR_DEFAULT = os.path.join(project_root, 'resources', 'all_frames')
    TRAIN_VAL_IMAGES_DIR_DEFAULT = os.path.join(project_root, 'resources', 'train_val_raw')
    
    # Файл будет создан в корневой директории проекта
    OUTPUT_FILENAME_DEFAULT = os.path.join(project_root, 'unselected_images_for_test.txt')

    unselected_frames, output_file_path = find_unselected_frames(
        all_images_dir=ALL_IMAGES_DIR_DEFAULT,
        train_val_images_dir=TRAIN_VAL_IMAGES_DIR_DEFAULT,
        output_filename=OUTPUT_FILENAME_DEFAULT
    )
    print(f"\nОбщее количество невыбранных кадров: {len(unselected_frames)}")
    print(f"Список сохранен в: {output_file_path}")