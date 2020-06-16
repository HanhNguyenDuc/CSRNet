from torchvision import datasets, transforms
import glob
from model import CSRNet
import torch
import cv2
import numpy as np
import PIL.Image as Image
import torchvision.transforms.functional as F
from config import *

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.458, 0.456, 0.456],
        std=[0.229, 0.224, 0.225]
    )
])

# map_location=torch.device('cpu')

root = 'data_csr/test_data/inputs'

img_paths = glob.glob(root + '/*')
model = CSRNet(FRONTEND_FEAT, BACKEND_FEAT)

checkpoint = torch.load('1model_best.pth (1).tar', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

mae = 0
img_paths.sort()
for i in range(len(img_paths)):
    print(img_paths[i])
    img_raw = cv2.imread(img_paths[i])
    cv2.imshow('raw_input', cv2.resize(img_raw, (img_raw.shape[1] // 8, img_raw.shape[0] // 8)))
    img = Image.open(img_paths[i]).convert('RGB')
    # print(cv2_img.shape)
    # img[0,:,:]=img[0,:,:]-92.8207477031
    # img[1,:,:]=img[1,:,:]-95.2757037428
    # img[2,:,:]=img[2,:,:]-104.877445883
    # img = img.cuda()
    # print(img)
    img = transform(img)
    output = model(img.unsqueeze(0))
    np_output = output.detach().numpy()
    # print(np_output)
    # print(img_raw.shape)
    np_output = np.transpose(np_output[0], (1, 2, 0))
    print(np_output.shape)
    # np_output = np_output.reshape((img_raw.shape[0] // 8, img_raw.shape[1] // 8))
    # np_output = cv2.resize(np_output, (img_raw.shape[1], img_raw.shape[0]), cv2.INTER_CUBIC)
    # visible_output = np.array(np_output, dtype='uint8')
    cv2.imshow('visible_output', np_output * 10.)
    print(np.sum(np_output))
    img_gt = cv2.imread(img_paths[i].replace('inputs', 'outputs'), 0)

    img_gt = cv2.resize(img_gt, (img_raw.shape[1] // 8, img_raw.shape[0] // 8), cv2.INTER_CUBIC)
    cv2.imshow('ground_truth', img_gt)

    cv2.waitKey(0) 
    # print(np.sum(img_gt / 255 / 255))

    