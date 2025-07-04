import cv2
from ultralytics import YOLO

# Загрузка модели
model = YOLO('best.pt')

video_path = 'video.mp4'  # путь к видео

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Ошибка открытия видео")
    exit()

paused = False
frame_pos = 0

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Видео завершилось")
            break
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Индекс текущего кадра (read уже сместил)

    else:
        # В режиме паузы устанавливаем позицию и считываем кадр
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            print("Ошибка при чтении кадра в паузе")
            break

    results = model(frame)
    annotated_frame = results[0].plot()

    if paused:
        info_text = "PAUSED | SPACE - play | <- / -> - rewind | Q / ESC - exit"
        color = (255, 255, 0)
    else:
        info_text = "SPACE - pause/play | <- / -> - rewind | Q / ESC - exit"
        color = (255, 50, 50)

    cv2.putText(annotated_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Detection", annotated_frame)

    key = cv2.waitKey(30)  # НЕ используем & 0xFF для стрелок

    if key == ord(' '):  # Пауза/плей
        paused = not paused

    elif key == 27 or key == ord('q'):  # ESC или q — выход
        break

    elif key == 81:  # Левая стрелка (перемотка назад)
        frame_pos = max(0, frame_pos - 30)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        paused = True

    elif key == 83:  # Правая стрелка (перемотка вперед)
        frame_pos = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, frame_pos + 30)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        paused = True

cap.release()
cv2.destroyAllWindows()
