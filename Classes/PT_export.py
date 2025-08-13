from ultralytics import YOLO

class PT2ONNX:
    def __init__(self, model_path: str, out_path: str = None):
        """
        初始化 PT2ONNX 类
        :param model_path: 输入模型路径
        :param out_path: 输出模型路径，默认为 None
        """
        self.model = YOLO(model_path)
        self.model.eval()  # 评估模型以确保其正确加载
    def export(self):
        """
        导出模型为 ONNX 格式
        """
        self.model.export(format='rknn',name="rk3588")
        return self.out_path