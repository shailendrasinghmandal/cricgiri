from ultralytics import YOLO

def main():

    model = YOLO("yolov8s.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=10,
        imgsz=1280,
        batch=8,
        device=0,
        workers=0
    )

if __name__ == "__main__":
    main()