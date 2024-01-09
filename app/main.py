from rembg import remove
import cv2
from keras.models import load_model
import pickle
import cv2
from fastai.vision.all import *
from typing import Union
from fastapi import FastAPI, APIRouter
import base64
from imageio import imread
from pydantic import BaseModel


import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

class BASE64_image(BaseModel):
    base64_image : str

def remove_background(img):
    img = cv2.imread('OIP.jpg')
    rmvd_img = remove(img)
    return rmvd_img

def ditector(img):
    #sample画像の前処理
    img = cv2.resize(img,dsize=(224,224))
    img = img.astype('float32')
    img /= 255.0
    img = img[None, ...]
    device = 'cpu'  # Check if CUDA is available, otherwise use CPU
    labels = ["cabbage", "carrot", "cucamber", "daikon", "eggplant", "green_onion", "onion", "potate", "satoimo", "shiitake", "spinach", "sweetpotate", "tomato", "turnip"]
    model = torch.load('model.pth', map_location=device)  # Load the model on the appropriate device
    model.eval()

    img = torch.from_numpy(img)
    print(img.size())
    img = img.transpose(1, 3)
    img = img.transpose(2, 3)
    # new_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    img = img.to(device)  # Move the image tensor to the same device as the model
    print(img.size())
    output = model(img)
    return labels[output.argmax(dim=1).item()]

app = FastAPI()
prefix_router = APIRouter(prefix="/vege-api/v1")



@prefix_router.get("/")
def read_root():
    return {"Hello": "World"}

@prefix_router.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@prefix_router.post("/detect_image")
def process_image(image: BASE64_image):
    # Base64文字列をデコードしてバイナリデータに変換
    image_data = base64.b64decode(image.base64_image)
    # バイナリデータをNumPy配列に変換
    nparr = np.frombuffer(image_data, np.uint8)
    # NumPy配列から画像を読み込む
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # print(img_tensor.size())
    # removed_bg = remove_background(img)
    # Detect object
    detected_object = ditector(img)
    return {"detected_object": detected_object}

app.include_router(prefix_router)