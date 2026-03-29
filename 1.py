from ultralytics import YOLO
from pathlib import Path
# 直接用修改后的 yaml 构建一个全新的空模型
model = YOLO('yolov8-cbam.yaml') 
current_file = Path(__file__).resolve()

# 2. 它的 parent 是 ultralytics 目录
# 3. 再往上一级 parent.parent 是 yolov8-first 目录
root_path = current_file.parent.parent

# 4. 拼接目标文件夹和文件名
data_yaml_path = root_path / "Minecraft.v8i.yolov8" / "data.yaml"
# 开始训练
results = model.train(
    data=str(data_yaml_path), 
    epochs=100, 
    imgsz=640, 
    device=0, # 或者 [0, 1] 如果你有多张卡
    workers=8,
    batch=16
)