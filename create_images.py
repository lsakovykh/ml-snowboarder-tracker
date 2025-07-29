import cv2
import os

video_path = 'resources/snowboard_day.mp4'
output_folder = 'resources/frames_for_labeling'

# Убедимся, что папка существует
os.makedirs(output_folder, exist_ok=True)
print(f"Папка для сохранения кадров: {os.path.abspath(output_folder)}")

cap = cv2.VideoCapture(video_path)

# Проверка, открылось ли видео
if not cap.isOpened():
    print(f"Ошибка: Не удалось открыть видеофайл по пути: {video_path}")
    exit() # Выходим, если видео не открылось

frame_rate = cap.get(cv2.CAP_PROP_FPS)
# Проверка, что frame_rate валиден
if frame_rate == 0:
    print("Ошибка: Частота кадров видео равна 0. Возможно, видеофайл поврежден или не поддерживается.")
    exit()

frame_interval = int(frame_rate * 1)
print(f"Частота кадров видео: {frame_rate} FPS")
print(f"Сохраняем каждый {frame_interval}-й кадр (примерно 1 кадр в секунду)")

count = 0
saved_frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Конец видеопотока или ошибка при чтении кадра.")
        break

    if count % frame_interval == 0:
        if frame is None:
            print(f"Предупреждение: Кадр {count} пуст (None), пропускаем сохранение.")
            count += 1 # Важно увеличить счетчик, чтобы не застрять
            continue

        # Добавим проверку размера кадра, чтобы убедиться, что он не пуст
        if frame.size == 0:
            print(f"Предупреждение: Кадр {count} имеет нулевой размер, пропускаем сохранение.")
            count += 1
            continue
        
        frame_filename = os.path.join(output_folder, f'frame_{saved_frame_count:04d}.jpg')
        
        # Попытка сохранения и проверка результата cv2.imwrite
        success = cv2.imwrite(frame_filename, frame)
        
        if success:
            saved_frame_count += 1
            print(f"Сохранен кадр: {frame_filename}")
        else:
            print(f"Ошибка: Не удалось сохранить кадр {frame_filename}. Проверьте путь и права доступа.")

    count += 1

cap.release()
print(f"Извлечено {saved_frame_count} кадров.")