from ultralytics import YOLO
import cv2
# Specify the video source and model path
model = YOLO('yolov8m.pt')  # Load the YOLOv8m model
source = 'bus.mp4'         # Path to your video file

# Create a window with a custom size (e.g., 800x600 pixels)
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection", 800, 600)  # Adjust width and height as needed

results = model.track(source=source, show=True, tracker='bytetrack.yaml')

# Read frames from the video
while results.video is not None:
    frame = results.frame
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
results.video.release()
cv2.destroyAllWindows()
