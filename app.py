import streamlit as st
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
from albumentations import (HorizontalFlip,VerticalFlip,CenterCrop, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise,RandomRotate90,Transpose,RandomBrightnessContrast,RandomCrop)
from albumentations.pytorch import ToTensor
from efficientnet_pytorch import EfficientNet
from PIL import Image
import cv2
import time

dog_dict = {0: 'affenpinscher',
            1: 'afghan_hound',
            2:  'african_hunting_dog',
            3:  'airedale',
            4:  'american_staffordshire_terrier',
            5:  'appenzeller',
            6:  'australian_terrier',
            7:  'basenji',
            8:  'basset',
            9:  'beagle',
            10:  'bedlington_terrier',
            11:  'bernese_mountain_dog',
            12:  'black-and-tan_coonhound',
            13:  'blenheim_spaniel',
            14:  'bloodhound',
            15:  'bluetick',
            16:  'border_collie',
            17:  'border_terrier',
            18:  'borzoi',
             19:  'boston_bull',
             20:  'bouvier_des_flandres',
             21:  'boxer',
             22:  'brabancon_griffon',
             23:  'briard',
             24:  'brittany_spaniel',
             25:  'bull_mastiff',
             26:  'cairn',
             27:  'cardigan',
             28:  'chesapeake_bay_retriever',
             29:  'chihuahua',
             30:  'chow',
             31:  'clumber',
             32:  'cocker_spaniel',
             33:  'collie',
             34:  'curly-coated_retriever',
             35:  'dandie_dinmont',
             36:  'dhole',
             37:  'dingo',
             38:  'doberman',
             39:  'english_foxhound',
             40:  'english_setter',
             41:  'english_springer',
             42:  'entlebucher',
             43:  'eskimo_dog',
             44:  'flat-coated_retriever',
             45:  'french_bulldog',
             46:  'german_shepherd',
             47:  'german_short-haired_pointer',
             48:  'giant_schnauzer',
             49:  'golden_retriever',
             50:  'gordon_setter',
             51:  'great_dane',
             52:  'great_pyrenees',
             53:  'greater_swiss_mountain_dog',
             54:  'groenendael',
             55:  'ibizan_hound',
             56:  'irish_setter',
             57:  'irish_terrier',
             58:  'irish_water_spaniel',
             59:  'irish_wolfhound',
             60:  'italian_greyhound',
             61:  'japanese_spaniel',
             62:  'keeshond',
             63:  'kelpie',
             64:  'kerry_blue_terrier',
             65:  'komondor',
             66:  'kuvasz',
             67:  'labrador_retriever',
             68:  'lakeland_terrier',
             69:  'leonberg',
             70:  'lhasa',
             71:  'malamute',
             72:  'malinois',
             73:  'maltese_dog',
             74:  'mexican_hairless',
             75:  'miniature_pinscher',
             76:  'miniature_poodle',
             77:  'miniature_schnauzer',
             78:  'newfoundland',
             79:  'norfolk_terrier',
             80:  'norwegian_elkhound',
             81:  'norwich_terrier',
             82:  'old_english_sheepdog',
             83:  'otterhound',
             84:  'papillon',
             85:  'pekinese',
             86:  'pembroke',
             87:  'pomeranian',
             88:  'pug',
             89:  'redbone',
             90:  'rhodesian_ridgeback',
             91:  'rottweiler',
             92:  'saint_bernard',
             93:  'saluki',
             94:  'samoyed',
             95:  'schipperke',
             96:  'scotch_terrier',
             97:  'scottish_deerhound',
             98:  'sealyham_terrier',
             99:  'shetland_sheepdog',
             100:  'shih-tzu',
             101:  'siberian_husky',
             102:  'silky_terrier',
             103:  'soft-coated_wheaten_terrier',
             104:  'staffordshire_bullterrier',
             105:  'standard_poodle',
             106:  'standard_schnauzer',
             107:  'sussex_spaniel',
             108:  'tibetan_mastiff',
             109:  'tibetan_terrier',
             110:  'toy_poodle',
             111:  'toy_terrier',
             112:  'vizsla',
             113:  'walker_hound',
             114:  'weimaraner',
             115:  'welsh_springer_spaniel',
             116:  'west_highland_white_terrier',
             117:  'whippet',
             118:  'wire-haired_fox_terrier',
             119:  'yorkshire_terrier' }

def transform(img_path,mean =(0.485, 0.456, 0.406) , std =(0.229, 0.224, 0.225)):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(256,256))
    list_transforms = []
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms(image = img)['image']

model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=120)
ckpt_path = "./dogbreed.pth"
device = torch.device("cpu")
model.to(device)
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"]) 
model.eval()

def predict(img_path):
    img = transform(img_path)
    img = img[None,:,:,:]
    pred = model(img.to(device)).detach().cpu().numpy()
    pred = np.argmax(pred,axis = 1)
    breed = dog_dict[pred[0]]
    return breed


st.title("Dog Breed Classification")

st.header("Dog Breed Classification Example")
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_person = st.file_uploader("Upload an image of the dog", type="jpg")


if uploaded_person is not None:
    img = Image.open(uploaded_person)
    st.image(img, caption="User Input", width=100, use_column_width=False)
    img.save("image.jpg")

if st.button('Execute'):
    img_path = "./image.jpg"
    st.write("Classifying...")
    breed = predict(img_path)
    execute_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        execute_bar.progress(percent_complete + 1)
    st.write('Woof Woof! I am a ',breed," !")

