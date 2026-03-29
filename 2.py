from ultralytics import YOLO

# 1. 加载你训练好的带 CBAM 的模型
model = YOLO('/kaggle/working/yolov8-first/runs/detect/train7/weights/best.pt')

# 2. 一键导出为 ONNX 格式 (opset=12 兼容性最好)
success = model.export(format='onnx', opset=12)

print("导出成功！去左侧文件栏下载 .onnx 文件吧！")