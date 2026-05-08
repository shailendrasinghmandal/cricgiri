import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train-4/weights/best.pt")

# Open input video
cap = cv2.VideoCapture("videos/test.mp4")

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create output video writer
out = cv2.VideoWriter(
    "output/detected_output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw detection results
    annotated_frame = results[0].plot()

    # Save frame
    out.write(annotated_frame)

    # Show live detection
    cv2.imshow("Detection", annotated_frame)

    # Press ESC to stop
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()