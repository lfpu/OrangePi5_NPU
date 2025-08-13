
from rknn.api import RKNN


class ExportModel:
    def __init__(self, model_path: str = None,out_path: str = None, dynamic_size: list = [1, 3, 112, 112]):
        """
        构造函数，可选传入模型路径
        """
        # 加载 InsightFace 模型
        self.out_path = out_path if out_path else 'model.rknn'
        self.model = RKNN()
        self.model.config(mean_values=[0, 0, 0], std_values=[128,128,128],dynamic_input=[[dynamic_size]] ,target_platform='rk3588')
        self.model.load_onnx(model_path)
        self.model.build(do_quantization=False)


    def export(self):
        self.model.export_rknn(self.out_path)
        self.model.release()