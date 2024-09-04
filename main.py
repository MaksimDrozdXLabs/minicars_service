from ultralytics import YOLO
import cv2

model_path = 'models/ryad288.pt'
video_path = 'res0.mp4'

model = YOLO(model_path)


if __name__ == '__main__':

    cap = cv2.VideoCapture(video_path)

    if isinstance(video_path, int):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, 250)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        pred = model.predict(frame, conf=0.3, imgsz=288, verbose=False)[0]
        boxes = pred.boxes.xywh[:, :2].tolist()[:2]
        img = pred.orig_img
        for (x, y) in boxes:
            img = cv2.circle(img, (int(x), int(y)), 20, (255, 0, 0), 10)
        cv2.imshow('drift', img)

        key = cv2.waitKey(1)
        if key == 27:  # if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break