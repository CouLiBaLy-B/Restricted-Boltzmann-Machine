import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

st.title("Image Generation using RBM")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Input Image", use_column_width=True)
    
    # Your model definition and training code here ...

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    img = transform(img)
    img = img.reshape(-1, 784).bernoulli()
    _, generated_img = rbm(img)

    # Show the generated image
    generated_img = generated_img.view(28, 28).detach().numpy()

    st.image(
        generated_img,
        caption="Generated Image",
        use_column_width=True,
        output_format="PNG",
    )
