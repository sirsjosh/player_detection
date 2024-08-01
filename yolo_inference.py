from ultralytics import YOLO

# Create a new YOLO model from scratch
# model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("/models/best.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.predict("./input_videos/08fd33_4.mp4", save=True)

print(results[0])
print('===============')
for box in results[0].boxes:
    print(box)
