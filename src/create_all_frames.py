import cv2
import os
from typing import Tuple # Импортируем Tuple для аннотации типов

def extract_frames_from_video(
    video_path: str,
    output_folder: str,
    frame_interval_seconds: int = 1 # Интервал сохранения кадров в секундах
) -> Tuple[int, str]:
    """
    Извлекает кадры из видеофайла и сохраняет их как изображения в указанную папку.
    Сохраняет один кадр за каждый заданный интервал времени.

    Args:
        video_path (str): Полный или относительный путь к исходному видеофайлу.
        output_folder (str): Путь к папке, куда будут сохраняться извлеченные кадры.
        frame_interval_seconds (int): Интервал в секундах, с которым будут сохраняться кадры.
                                      Например, 1 означает сохранение ~1 кадра в секунду.

    Returns:
        Tuple[int, str]: Кортеж, содержащий количество сохраненных кадров и
                         абсолютный путь к папке сохранения.
    """
    # Убедимся, что папка для сохранения кадров существует
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    # full_output_path = os.path.join(project_root, output_folder) # Если output_folder указан относительно корня проекта
    # Для текущей структуры, где output_folder уже относительный к рабочей директории
    os.makedirs(output_folder, exist_ok=True)
    abs_output_path = os.path.abspath(output_folder)
    print(f"Папка для сохранения кадров: {abs_output_path}")

    # Инициализация объекта VideoCapture
    cap = cv2.VideoCapture(video_path)

    # Проверка, открылось ли видео
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл по пути: {video_path}")
        return 0, abs_output_path # Возвращаем 0 кадров при ошибке

    # Получаем частоту кадров видео
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate <= 0: # Проверка, что frame_rate валиден и не равен 0
        print("Ошибка: Частота кадров видео равна 0 или некорректна. Возможно, видеофайл поврежден или не поддерживается.")
        cap.release()
        return 0, abs_output_path

    # Вычисляем интервал кадров для сохранения
    # Например, если FPS=30, interval_frames=30, сохраняем каждый 30-й кадр (1 кадр в секунду)
    frames_to_skip = int(frame_rate * frame_interval_seconds)
    if frames_to_skip < 1: # Убедимся, что мы пропускаем хотя бы 1 кадр, если interval_seconds очень мал
        frames_to_skip = 1 
    
    print(f"Частота кадров видео: {frame_rate:.2f} FPS")
    print(f"Сохраняем каждый {frames_to_skip}-й кадр (примерно {frame_interval_seconds} кадр(а) в секунду)")

    count = 0 # Общий счетчик прочитанных кадров
    saved_frame_count = 0 # Счетчике сохраненных кадров

    while True:
        ret, frame = cap.read() # Читаем следующий кадр

        # Если кадры закончились или произошла ошибка чтения
        if not ret:
            print("Конец видеопотока или ошибка при чтении кадра.")
            break

        # Проверяем, нужно ли сохранить текущий кадр
        if count % frames_to_skip == 0:
            # Добавим более надежные проверки на пустоту/корректность кадра
            if frame is None or frame.size == 0:
                print(f"Предупреждение: Кадр {count} пуст или имеет нулевой размер. Пропускаем сохранение.")
                # Если кадр пуст, все равно инкрементируем основной счетчик и продолжаем
                count += 1
                continue
            
            # Формируем имя файла кадра с нумерацией
            frame_filename = os.path.join(output_folder, f'frame_{saved_frame_count:04d}.jpg')
            
            # Попытка сохранения и проверка результата cv2.imwrite
            success = cv2.imwrite(frame_filename, frame)
            
            if success:
                saved_frame_count += 1
                # Отключено для уменьшения вывода при большом количестве кадров
                # print(f"Сохранен кадр: {frame_filename}") 
            else:
                print(f"Ошибка: Не удалось сохранить кадр {frame_filename}. Проверьте путь и права доступа.")

        count += 1 # Всегда инкрементируем общий счетчик кадров

    cap.release() # Освобождаем ресурсы видеопотока
    print(f"Извлечено и сохранено {saved_frame_count} кадров.")
    return saved_frame_count, abs_output_path

if __name__ == "__main__":
    # Эти параметры могут быть настроены при запуске скрипта напрямую
    # Если вы вызываете это из ноутбука, убедитесь, что пути заданы там
    video_input_path = '../resources/snowboard_day.mp4' # Путь к исходному видео
    frames_output_folder = '../resources/all_frames' # Папка для сохранения кадров

    # Вызываем функцию
    num_saved_frames, output_dir_path = extract_frames_from_video(
        video_path=video_input_path,
        output_folder=frames_output_folder,
        frame_interval_seconds=1
    )
    print(f"Всего сохранено кадров: {num_saved_frames} в директории: {output_dir_path}")