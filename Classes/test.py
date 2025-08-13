import cv2
import os
from insightface.app import FaceAnalysis
import time
import onnxruntime as ort

print(ort.get_device())

model = FaceAnalysis("buffalo_s",'./', provider=['CUDAExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

filepath = "C:\\Users\\lfpu\\OneDrive - kochind.com\\Pictures\\Camera Roll\\"
files = os.listdir(filepath)
for file in files:
    if file.endswith(".jpg") or file.endswith(".png"):
        img=cv2.imread(os.path.join(filepath, file))
        if img is not None:
            t= time.time()
            faces= model.get(img)
            if faces:
                print(f"Detected faces in {file}: {len(faces)}, ms: {time.time() - t:.2f}")
                for i, face in enumerate(faces):
                    print(f"Face {i+1}: {face.embedding}")
        else:
            print(f"Failed to read {file}")
