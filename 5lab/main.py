import cv2 as cv
import numpy as np

# 1) Зчитати кольорове зображення з файлу як numpy масив img1 (imread)
img1 = cv.imread('image.jpg')

# 2) У зображення img1 вставити:
# 2.1) Прямокутник зеленого кольору (BGR) з товщиною лінії 1
green_rect_x1, green_rect_y1 = 50, 50
green_rect_x2, green_rect_y2 = 200, 150
cv.rectangle(img1, (green_rect_x1, green_rect_y1), (green_rect_x2, green_rect_y2),
             color=(0, 255, 0), thickness=1)

# 2.2) Прямокутник синього кольору з товщиною лінії 2
blue_rect_x1, blue_rect_y1 = 250, 100
blue_rect_x2, blue_rect_y2 = 400, 200
cv.rectangle(img1, (blue_rect_x1, blue_rect_y1), (blue_rect_x2, blue_rect_y2),
             color=(255, 0, 0), thickness=2)

# 2.3) Текст червоного кольору. Розмір шрифту 3, товщина 2
cv.putText(img1, "OpenCV Test", (100, 300),
           fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=3,
           thickness=2, color=(0, 0, 255))

# 2.4) Вивести отримане зображення у вікно
cv.imshow('Original with drawings', img1)

# 3) Перетворити img1 у зображення у градаціях сірого img2
img2 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

# 4) Перетворити img1 у зображення у форматі LAB img3
img3 = cv.cvtColor(img1, cv.COLOR_BGR2LAB)

# 5) З img3 отримати лише L канал як масив img4 (використати зріз)
img4 = img3[:, :, 0]  # L канал - перший канал в LAB

# 6) Вивести зображення img2 і img4 у вікна
cv.imshow('Grayscale (img2)', img2)
cv.imshow('L channel (img4)', img4)
cv.waitKey(5000)  # Очікувати натискання клавіші 5 секунд

# 7) Вирізати частини з прямокутників і створити нове зображення img5
# Вирізати частину з зеленого прямокутника
green_rect_part = img1[green_rect_y1:green_rect_y2, green_rect_x1:green_rect_x2].copy()
# Вирізати частину з синього прямокутника
blue_rect_part = img1[blue_rect_y1:blue_rect_y2, blue_rect_x1:blue_rect_x2].copy()

# Створити img5, що містить обидві вирізані частини розміщені одна під одною
# Спочатку визначимо розмір нового зображення
height1 = green_rect_part.shape[0]
width1 = green_rect_part.shape[1]
height2 = blue_rect_part.shape[0]
width2 = blue_rect_part.shape[1]
max_width = max(width1, width2)
total_height = height1 + height2

# Створити пусте зображення потрібного розміру
img5 = np.zeros((total_height, max_width, 3), dtype=np.uint8)
# Розмістити зелений прямокутник зверху
img5[:height1, :width1] = green_rect_part
# Розмістити синій прямокутник знизу
img5[height1:, :width2] = blue_rect_part

# Вивести img5 у вікно
cv.imshow('Combined rectangles (img5)', img5)

# 8) Нормувати зображення img1, отримавши numpy масив img6 (значення в діапазоні [0, 1])
img6 = img1.astype(np.float32) / 255.0

# 9) З img6 отримати зображення img7, виконавши обернене нормування (значення в діапазоні [0, 255])
img7 = img6 * 255
# Перетворення значень у цілі
img7 = img7.astype(np.uint8)

# 10) Вивести зображення img7 у вікно і зберегти у файл
cv.imshow('Renormalized image (img7)', img7)
cv.imwrite('output_image.jpg', img7)

# 11) Збільшити розмір зображення img7 у 2 рази по x і у 3 рази по y
h, w = img7.shape[:2]
img7_resized = cv.resize(img7, (w * 2, h * 3))
cv.imshow('Resized img7', img7_resized)

# 12) Перетворити LAB зображення img3 у зображення img8 у форматі BGR
img8 = cv.cvtColor(img3, cv.COLOR_LAB2BGR)
cv.imshow('Back to BGR (img8)', img8)

# 13) Побудувати замкнений багатокутник на зображення img1 та вивести у вікно
vertices = np.array([[100, 400], [200, 300], [300, 350], [250, 450]], np.int32)
vertices = vertices.reshape((-1, 1, 2))
cv.polylines(img1, [vertices], isClosed=True, color=(0, 255, 255), thickness=2)
cv.imshow('Image with polygon', img1)

# 14) Визначити, чи точка лежить всередині багатокутника
test_point = (200, 350)  # Замініть на потрібні координати
result = cv.pointPolygonTest(vertices, test_point, False)
if result > 0:
    print(f"Точка {test_point} лежить всередині багатокутника")
else:
    print(f"Точка {test_point} лежить ззовні багатокутника")


# 15) Вивести у консоль координати курсору мишки
def mouse_coords(event, x, y, flags, params):
    if event == cv.EVENT_MOUSEMOVE:
        print(f"Координати курсору: x={x}, y={y}")


cv.namedWindow('Mouse coordinates')
cv.setMouseCallback('Mouse coordinates', mouse_coords)
cv.imshow('Mouse coordinates', img1)


# 16) Зчитати відео з файлу та відобразити його у вікні
def play_video(video_path):
    cap = cv.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv.imshow('Video', frame)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()


# Замініть 'input_video.mp4' на шлях до вашого відео
# play_video('video.mp4')

# 17) Зчитати відео з файлу і намалювати прямокутник на кожному кадрі
def play_video_with_rectangle(video_path):
    cap = cv.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Малюємо прямокутник на кожному кадрі
        cv.rectangle(frame, (50, 50), (200, 150), (0, 255, 0), 2)

        cv.imshow('Video with Rectangle', frame)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()


# Замініть 'input_video.mp4' на шлях до вашого відео
# play_video_with_rectangle('video.mp4')

# Очікування натискання клавіші перед закриттям усіх вікон
cv.waitKey(0)
cv.destroyAllWindows()