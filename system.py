import numpy as np
import streamlit as st
import torch
import torchvision
from PIL import Image


def classify(model, img):
  transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]
                )]
              )
  img = Image.open(img)
  img = transform(img)[None,]
  predict = model(img).detach()
  predict = torch.argmax(predict)

  img_class = {
    0: 'Mask weared incorrect',
    1: 'With mask',
    2: 'Without mask'
  }

  return img_class[int(predict.item())]

def main(model):
  st.title('INF492 - Visão Computacional')

  img = st.file_uploader('Upload a image', ['jpg', 'png', 'jpeg'], accept_multiple_files=False)

  img_class = ''

  if img is not None:
    img_class = classify(model, img)

  if img is None:
    img = np.asarray(Image.open('placeholder.png'))

  st.image(img)
  st.markdown('### ' + img_class)



if __name__ == '__main__':
  model = torch.load('models/Net-mask-98.14.pkl')
  model.eval()
  model.to('cpu')
  main(model)
