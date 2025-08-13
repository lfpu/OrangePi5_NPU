import os
from fastapi import FastAPI, File, UploadFile,Request
from fastapi.responses import FileResponse
import cv2
import numpy as np
from Classes.faceOP import FaceOperation
from Classes.exportmodel import ExportModel
from Classes.findface import FindFace
from Classes.apiobj import CompareFace
import base64

if not os.path.exists('/app/Data/models'):
    os.makedirs('/app/Data/models')

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, RKNN!"}

@app.post("/Model/ExportRKNN/")
async def export_model_endpoint(file: UploadFile = File(...),dynamicSize: list = [1,3,112, 112]):
    # 检查上传的文件类型
    if not file.filename.endswith('.onnx'):
        return {"error": "Only ONNX model files are supported."}
    # 读取上传的模型文件
    model_bytes = await file.read()
    model_path = '/app/Data/models/model.onnx'  # 假设模型文件名为'model.onnx'
    with open(model_path, 'wb') as f:
        f.write(model_bytes)
    out_path = '/app/Data/models/model.rknn'
    recognizer =ExportModel(model_path=model_path, out_path=out_path, dynamic_size=dynamicSize)
    recognizer.export()
    # 返回导出后的模型文件
    return FileResponse(path=out_path, filename='model.rknn', media_type='application/octet-stream')    
@app.post("/Face/testRcong/")
async def register_face_endpoint(img: UploadFile = File(...)):
    # 读取上传的图片字节
    image_bytes = await img.read()
    if not image_bytes:
        return {"error": "No image data provided"}
    recognizer = FaceOperation('/app/Data/models/facefea.rknn')
    result =recognizer.register_face(image_bytes, size=112)
    recognizer.release()
    return result
@app.post("/Face/testFind/")
async def find_face_endpoint(img: UploadFile = File(...),size: int = 224):
    imgpath= '/app/Data/models/test.jpg'
    # 保存上传的图片
    with open (imgpath, "wb") as f:
        f.write(await img.read())
    find_face = FindFace('/app/Data/models/facefind.rknn')
    # 假设 find_faces 方法接收图片字节并返回人脸区域
    face_boxes = find_face.find_faces(imgpath, size=size)
    if not face_boxes or len(face_boxes) == 0:
        return {"error": "No faces found in the image"}
    find_face.release()
    encodelist=[base64.b64encode(face_box).decode('utf-8') for face_box in face_boxes]
    return encodelist  # 返回人脸区域的坐标列表
@app.post("/Face/Recognition/")
async def recognition_endpoint(request: Request):

    # 读取上传的图片字节
    image_bytes = await request.body()
    if not image_bytes:
        return {"error": "No image data provided"}
    recognizer = FaceOperation('/app/Data/models/facefea.rknn')  # 假设模型路径为'/models/reconModel.onnx'
    # 假设 recognition 方法接收图片字节并进行人脸注册
    result = recognizer.register_face(image_bytes, size=112)
    recognizer.release()
    return result
@app.post("/Face/FindFace")
async def find_face(request: Request):
    # 读取上传的图片字节
    image_bytes = await request.body()
    if not image_bytes:
        return {"error": "No image data provided"}
    imgpath = '/app/Data/models/face.jpg'
    with open(imgpath, "wb") as f:
        f.write(image_bytes)
    find_face = FindFace('/app/Data/models/facefind.rknn')
    # 假设 find_faces 方法接收图片路径并返回人脸区域
    face_boxes = find_face.find_faces(imgpath, size=640)
    find_face.release()
    if not face_boxes or len(face_boxes) == 0:
        return {"error": "No faces found in the image"}
    encodelist=[base64.b64encode(face_box).decode('utf-8') for face_box in face_boxes]
    return encodelist

@app.post("/Face/Compare/")
async def compare_face_endpoint(compare_face: CompareFace):
    
    compare = FaceOperation('/app/Data/models/facefea.rknn')
    find_face = FindFace('/app/Data/models/facefind.rknn')
    results = []
    for imgpath in compare_face.imgpaths:
        # 假设 find_faces 方法接收图片路径并返回人脸区域x
        face_boxes = find_face.find_faces(imgpath, size=compare_face.findsize)
        if not face_boxes or len(face_boxes) == 0:
            continue
        for face_box in face_boxes:
            embing = compare.register_face(face_box, size=compare_face.comparesize)
            faceid=0
            for checkface in compare_face.checkfaces:
                result = compare.compare_face(embing, checkface)
                if result is not None:
                    results.append({
                        "faceid": faceid,
                        "imgpath": imgpath,
                        "similarity": result
                    })
                faceid += 1
    find_face.release()
    compare.release()
    return {"results": results}
