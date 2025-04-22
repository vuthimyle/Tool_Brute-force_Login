import requests
from bs4 import BeautifulSoup
import time
import os
from requests.adapters import HTTPAdapter
import torch
from train_captcha_model_logon import create_model, predict_captcha 

def solve_captcha(image_bytes, attempt_number=0):
    try:
        # Load model with same parameters as train_captcha_model_logon.py
        img_width = 160
        img_height = 60
        num_characters = 4
        char_list = '0123456789'
        
        model = create_model(img_width, img_height, num_characters, char_list)
        model.load_state_dict(torch.load('best_model_32.pth', map_location=torch.device('cuda')))  # Changed model file
        model.eval()
        
        # Predict
        captcha_text = predict_captcha(model, image_bytes, char_list)
        print(f"Predicted CAPTCHA: {captcha_text}")
        
        
        return captcha_text
    except Exception as e:
        print(f"Error solving captcha: {e}")
        return None



