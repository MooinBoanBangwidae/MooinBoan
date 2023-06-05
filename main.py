import cv2
import pygame

pygame.mixer.init()

cam = cv2.VideoCapture(1)

while cam.isOpened():
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        pygame.mixer.music.load('beep.wav')  # 소리 파일 경로
        pygame.mixer.music.play()
        pygame.time.wait(200)  # 소리 재생 시간 (밀리초)

    if cv2.waitKey(10) == ord('q'):
        break
    cv2.imshow('Granny Cam', frame1)

pygame.mixer.quit()