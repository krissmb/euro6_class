import cv2
from ultralytics import YOLO

# Загрузить модель
model = YOLO('best.pt')  # путь к твоей модели

# Путь к видеофайлу
video_path = 'video.mp4'

# Открываем видео
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка открытия видеофайла")
    exit()

paused = False
frame_pos = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Достигнут конец видео или ошибка чтения")
            break
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    else:
        # Если на паузе, просто получаем текущий кадр
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            break

    # Выполняем детекцию
    results = model(frame)

    # Отрисовка результатов на кадре (YOLOv8 умеет рисовать сам)
    annotated_frame = results[0].plot()

    cv2.putText(annotated_frame, f'Frame: {frame_pos}/{total_frames}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('YOLOv8 Detection', annotated_frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        # Выход по q
        break
    elif key == ord('p'):
        # Пауза / продолжение по p
        paused = not paused
    elif key == ord('d'):
        # Перемотка вперёд на 30 кадров (~1 секунда при 30fps)
        frame_pos = min(frame_pos + 30, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    elif key == ord('a'):
        # Перемотка назад на 30 кадров
        frame_pos = max(frame_pos - 30, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

cap.release()
cv2.destroyAllWindows()
