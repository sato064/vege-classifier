from keras.models import load_model
import pickle
import cv2
from fastai.vision.all import *

#sample画像の前処理
img = cv2.imread('rmvd.png')
img = cv2.resize(img,dsize=(224,224))
img = img.astype('float32')
img /= 255.0
img = img[None, ...]

model = torch.load('model.pth')
model.eval()

output = model(img)
print(output.argmax(dim=1))

