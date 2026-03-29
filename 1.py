from ultralytics import YOLO
from pathlib import Path
# 直接用修改后的 yaml 构建一个全新的空模型
model = YOLO('yolov8-cbam.yaml') 
current_file = Path(__file__).resolve()
target_file = current_file.parent.parent / "Minecraft.v8i.yolov8" / "data.yaml"
# 开始训练
results = model.train(
    data=str(target_file), 
    epochs=100, 
    imgsz=640, 
    device=0, # 或者 [0, 1] 如果你有多张卡
    workers=8,
    batch=16
)