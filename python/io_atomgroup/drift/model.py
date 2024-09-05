


class Model:
    def __init__(self):
        self.model_path = 'models/ryad288.pt'
        self.video_path = 'res0.mp4'

    def model(self):
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)

    def process(
        self,
        frame_get,
        frame_set,
        frame_cv,
    ):
        #import cv2
        #cap = cv2.VideoCapture(self.video_path)

        #if isinstance(self.video_path, int):
        #    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        #    cap.set(cv2.CAP_PROP_EXPOSURE, 250)
        #    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
        #    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        #    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        #    cap.set(cv2.CAP_PROP_FPS, 30)

        while True:
            with frame_cv:
                frame_cv.wait()

                frame4 = frame_get()
                frame = cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB),

            # ret, frame = cap.read()

            # if not ret:
            #    continue

            pred = self.model.predict(frame, conf=0.3, imgsz=288, verbose=False)[0]
            boxes = pred.boxes.xywh[:, :2].tolist()[:2]
            img = pred.orig_img
            for (x, y) in boxes:
                img = cv2.circle(img, (int(x), int(y)), 20, (255, 0, 0), 10)

            # cv2.imshow('drift', img)

            with frame_cv:
                frame_set(img)

            #key = cv2.waitKey(1)
            #if key == 27:  # if ESC is pressed, exit loop
            #    cv2.destroyAllWindows()
            #    break

    @classmethod
    def run(cls):
        m = cls()
        m.model()
        m.process()


if __name__ == '__main__':
    Model.run()
