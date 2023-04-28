import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import streamlit as st
from torchvision.utils import make_grid

class RBM(nn.Module):
   def __init__(self,
               n_vis=784,
               n_hin=500,
               k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k
    
   def sample_from_p(self,p):
       return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))
    
   def v_to_h(self,v):
        p_h = torch.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h
    
   def h_to_v(self,h):
        p_v = torch.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v
        
   def forward(self,v):
        pre_h1,h1 = self.v_to_h(v)
        
        h_ = h1
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)
        
        return v,v_
    
   def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()


# Load the pre-trained model and weights
rbm = RBM(k=1)
rbm.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
rbm.eval()

# Function to generate an image
def generate_image():
    noise = torch.randn(1, 784)
    with torch.no_grad():
        _, generated_image = rbm(noise)
    generated_image = generated_image.view(1, 1, 28, 28)
    return generated_image

# Streamlit app
def main():
    st.header("Image Generation with RBM")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = plt.imread(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Generate Image"):
            generated_image = generate_image()
            generated_image_np = np.transpose(generated_image.squeeze().numpy(), (2, 1, 0))

            st.image(generated_image_np, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()
