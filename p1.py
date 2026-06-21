from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")  

file = input("Enter the filename: ").strip()

if not os.path.exists(file):
    print("File not found!!")
else:
    results = model.predict(
        source = file,
        save = True,
        project="runs/detect",
        name="output",
        exist_ok=True
    )

    print('Detection completed')
    print(f"Result saved to: runs/detect/output/{file}")

    for box in results[0].boxes :
        print(f"{model.names[int(box.cls)]}-{float(box.conf):.0%}")