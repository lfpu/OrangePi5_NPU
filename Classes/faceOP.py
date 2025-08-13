import numpy as np
import cv2
from rknn.api import RKNN

class FaceOperation:
    def __init__(self, model_path: str = None):
        """
        构造函数，可选传入模型路径
        """
        # 加载 InsightFace 模型
        self.model= RKNN()
        self.model.load_rknn(model_path)
        self.model.init_runtime(target='rk3588')
    def register_face(self, image_bytes: bytes,size=112):
        """
        人脸注册，返回特征向量
        :param image_bytes: 输入图片的字节流（bytes）
        :return: 特征向量（list），未检测到人脸返回None
        """
        # 将字节流解码为numpy.ndarray（BGR格式）
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.resize(image, (size, size))
        image =image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        outputs=self.model.inference(inputs=[image],data_format='nchw')
        embedding = outputs[0][0]
        return embedding.tolist()
    def compare_face(self, embedding1: list, embedding2: list):
        """
        比较两个人脸特征向量，返回相似度
        :param embedding1: 第一个人脸的特征向量
        :param embedding2: 第二个人脸的特征向量
        :param threshold: 相似度阈值，默认0.5
        :return: 相似度（float），如果相似度大于阈值则认为是同一人
        """
        if not embedding1 or not embedding2:
            return None
        # 计算余弦相似度
        norm1= np.linalg.norm(embedding1)
        norm2= np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return None
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)


        return similarity
    def release(self):
        """
        释放模型资源
        """
        self.model.release()