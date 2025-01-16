import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from PIL import Image

model_path ='models/RRDB_ESRGAN_x4.pth' #'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
#device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
device = torch.device('cpu')

test_img_folder = 'LR/*'
US_folder='results/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    j=0
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
    for path1 in glob.glob(US_folder):
        j=j+1
        if(idx==j):
           img_1=cv2.imread(path)
           img_2=cv2.imread(path1)
           image = Image.open(path1)
           width, height = image.size
           img_1=cv2.resize(img_1,(width,height))
           img_2=cv2.resize(img_2,(width,height))
           #final_img=cv2.add(img_1,img_2)
           final_img=cv2.addWeighted(img_1,0.4,img_2,0.6,0)
           cv2.imwrite('final/{:s}_rlt1.png'.format(base), final_img)
           #cv2.imshow("img1",final_img)
           #cv2.waitKey(0)
           #cv2.destroyAllWindows()
        
        #final_img=cv2.addWeighted(img_1,0.9,img_2,0.1,0)
        #parameters=1st_img, weight_of_img1,2nd_img,weight_of_img2
        #cv2.imshow("final",final_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
'''img_1=cv2.imread("G:\ESRGAN\LR\baboon.png")
img_2=cv2.imread("G:\ESRGAN\results\baboon_rlt.png")
image = Image.open("G:\ESRGAN\results\baboon_rlt.png")
width, height = image.size
img_1=cv2.resize(img_1,(width,height))
final_img=cv2.add(img_1,img_2)
cv2.imshow("final",final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

'''idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    #print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
print("number of images = " + idx)
for i in range(0,idx):'''

