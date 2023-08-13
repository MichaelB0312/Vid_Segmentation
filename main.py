import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
# pytorch
import torch
import torch.nn as nn
import torchvision
# import datasets in torchvision
import torchvision.datasets as datasets
# import model zoo in torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import utils

# section1
import os
from PIL import Image


#%cd '/content/drive/MyDrive/code'
from frame_video_convert import video_to_image_seq

#extecting frames
video_to_image_seq('./data/my_data/orig_vdo.mp4')

frames_dir = './datasets/OTB/img/Custom'
print(len(os.listdir(frames_dir)))

#make preprocessing
frames_dict = {}
cnt = 0
for image in (os.listdir(frames_dir)):
  img = Image.open(os.path.join(frames_dir,image))
  transforms.ToPILImage(img)
  resize = transforms.Resize((1280,720))
  resized_img = resize(img)
  frames_dict[cnt] = resized_img
  cnt = cnt + 1

#show two figures (first and last frame)
plt.figure(figsize=(10,15))
plt.subplot(1,2,1)
plt.imshow(frames_dict.get(0))

plt.subplot(1,2,2)
plt.imshow(frames_dict.get(len(os.listdir(frames_dir))-1))
plt.show()

# section 2
import cv2


from ipynb.fs.full.part2 import decode_segmap

# define device
# check if there is a GPU available
print(torch.cuda.is_available())
# check what is the current available device
if torch.cuda.is_available():
 print("current device: ", torch.cuda.current_device())
# automatically choose device
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # use gpu 0 if it is available,
#o.w. use the cpu
print("device: ", dev)

# segmentation using deep method with deep lab v3, same way as in ex. 2.6
dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
# for a full explaination of the algorithm, see ex 2.5 code
def segment(net, img, show_orig=True , dev='cuda'):
  if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
  # Comment the Resize and CenterCrop for better inference results
  trf = transforms.Compose([transforms.Resize(640),
                   #T.CenterCrop(224),
                   transforms.ToTensor(),
                   transforms.Normalize(mean = [0.485, 0.456, 0.406],
                               std = [0.229, 0.224, 0.225])])
  inp = trf(img).unsqueeze(0).to(dev)
  out = net.to(dev)(inp)['out']
  max_label = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  rgb = decode_segmap(max_label)
  return rgb
  #plt.imshow(rgb); plt.axis('off'); plt.show()

segmented_frames = {}
#the prcentage
background_pixels_num = np.zeros(len(frames_dict))
for cnt,image in enumerate(frames_dict):
  segmented_frames[cnt] = segment(dlab, frames_dict.get(image), show_orig=False)
  # check where did the network fail. Too much background pixels mean our object
  # has been sliced.
  background_pixels_num[cnt] = (np.sum(segmented_frames[cnt] == (0,0,0)))
  norm = segmented_frames[0].shape[0]*segmented_frames[0].shape[1]*segmented_frames[0].shape[2]
  background_pixels_num[cnt] = background_pixels_num[cnt]/norm
  #if(background_pixels_num[cnt] > 0.9):
  #  print(cnt)