from pydantic import BaseModel
class CompareFace(BaseModel):
    imgpaths:list[str]
    # 假设 imgpaths 是一个字符串列表，包含图片的路径或字节流
    checkfaces:list[list[float]] = None
    findsize:int = 640
    comparesize:int = 112