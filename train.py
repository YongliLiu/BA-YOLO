from ultralytics import YOLO

def r():
    # 加载模型

    model = YOLO(r"E:\liuyongli\code\yolov8\ultralytics\cfg\models\v8\yolov8s-AM2.yaml")
    # model._load(r"C:\Users\HP\Desktop\ultralytics-main\yolov8m.pt")  #yolov8n.pt 加载预训练模型（推荐用于训练）

    # Use the model
    results = model.train(data=r"E:\liuyongli\code\yolov8\ultralytics\cfg\datasets\coco.yaml", epochs=200, batch=8, workers=8, resume=True, name='DIOR-2GSA')
    results = model.val()  # 在验证集上评估模型性能ultralytics/models/v8/yolov8x.yaml
    # results = model("https://ultralytics.com/images/bus.jpg")  # 预测图像
    # success = model.export(format="onnx")

    # results = model.predict(source="E:/liuyongli/datasets/HRRSD/images/1", save=True, save_txt=True, name='test-H-all')



if __name__ == "__main__":
    r()
