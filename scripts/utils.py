import os
import re

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
    # Если скрипт запускается из ноутбука (папка notebooks/),
    # то project_dir должен быть относительным к корневой папке проекта.
    # Поэтому мы используем os.path.join, чтобы правильно построить путь.
    
    # Получаем текущую рабочую директорию (обычно папка notebooks/)
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) # Получаем путь к scripts/
    
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