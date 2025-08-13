import numpy as np
import cv2
from rknn.api import RKNN


class FindFace:
    def __init__(self, model_path: str = None):
        """
        构造函数，可选传入模型路径
        """
        self.model = RKNN()
        self.model.load_rknn(model_path)
        self.model.init_runtime(target='rk3588')
    def xywh2xyxy(self, x):
        """
        将xywh格式的框转换为xyxy格式
        :param boxes: 输入框的坐标，格式为[x, y, w, h]
        :return: 转换后的坐标，格式为[x1, y1, x
        x2, y2]
        """
        y=np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y
    def nms_boxes(self, boxes, scores, iou_threshold=0.45):
        """
        非极大值抑制
        :param boxes: 输入框的坐标，格式为[x1, y1, x2, y2]
        :param scores: 每个框的置信度分数
        :param iou_threshold: IOU阈值，默认0.45
        :return: 保留的框的索引
        """
        x= boxes[:, 0]
        y= boxes[:, 1]
        w= boxes[:, 2] - x
        h= boxes[:, 3] - y
        areas = w * h
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            w1 = np.maximum(0.0, xx2 - xx1+0.00001)
            h1 = np.maximum(0.0, yy2 - yy1+0.00001)
            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        keep= np.array(keep, dtype=np.int32)
        return keep

    def find_faces(self, imgpath: str, size=112):
        """
        人脸检测，返回检测到的人脸区域
        :param image_bytes: 输入图片的字节流（bytes）
        :return: 检测到的人脸区域（list），未检测到人脸返回None
        """

        Orgimage = cv2.imread(imgpath)
        if Orgimage is None:
            return None
        image = cv2.cvtColor(Orgimage, cv2.COLOR_BGR2RGB)
        scaleX= image.shape[1] / size
        scaleY= image.shape[0] / size
        image = cv2.resize(image, (size, size))
        # image = image.transpose((2, 0, 1))
        # image = np.expand_dims(image, axis=0)
        # image = image.astype(np.float32)
        
        outputs = self.model.inference(inputs=[image])
        
            # 后处理
        output = outputs[0]  # shape: [1, 5, N] 或 [5, N]
        if output.ndim == 3:
            output = output[0]  # 去掉 batch 维度
        # 假设输出是人脸区域的坐标

        boxes =output[:4,:].T 
        scores = output[4,:]

        mask = scores >= 0.3  # 假设阈值为0.5
        boxes = boxes[mask]
        scores = scores[mask]
        classess=np.zeros_like(scores,dtype=int)

        #坐标转换,不做转换
        boxes=self.xywh2xyxy(boxes)

        #NMS非极大值抑制
        keep=self.nms_boxes(boxes, scores, 0.45)
        boxes = boxes[keep]
        scores = scores[keep]
        classess = classess[keep]
        byts=[]
        #img_1=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if boxes is not None and len(boxes) > 0:
            #按照boxes的坐标裁剪图像，转成bytes
            boxes[:, 0] = np.clip(boxes[:, 0] * scaleX, 0, Orgimage.shape[1])
            boxes[:, 1] = np.clip(boxes[:, 1] * scaleY, 0, Orgimage.shape[0])
            boxes[:, 2] = np.clip(boxes[:, 2] * scaleX, 0, Orgimage.shape[1])
            boxes[:, 3] = np.clip(boxes[:, 3] * scaleY, 0, Orgimage.shape[0])
            boxes = boxes.astype(int)
            img = Orgimage[int(boxes[0][1]):int(boxes[0][3]), int(boxes[0][0]):int(boxes[0][2])]
            img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
            byts.append(img_bytes)
        return byts
    def release(self):
        """
        释放模型资源
        """
        self.model.release()