import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules1_4 import UNet
import logging
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import dumpman
import cv2
from skimage.color import lab2rgb
from skimage import color, io

from skimage.transform import resize

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

X = dumpman.undumper('thermal.pkl') #for training only
z = dumpman.undumper('rgb.pkl')

X_train, X_rem, z_train, z_rem = train_test_split(X, z, test_size=0.05, random_state=42)


X_valid, X_test, z_valid, z_test = train_test_split(X_rem, z_rem, test_size=0.5, random_state=42)

batch = 64
img_row,img_col = 32, 32
img_size = 32

print("Number of testing samples: " + str(len(X_test)))

print(X_test[50])
print(z_test[50])

fname = './fused_3_CE'

class_index = [0,1]
overlay = [0,255]

def label_to_mask(mask):
    label = np.zeros((mask.shape[1],mask.shape[2]), dtype=np.uint8)
    
    for i, class_val in enumerate(class_index):
            bool_mask = np.array(
                (mask[i] == 1),
                dtype=bool,
            )           
            label[bool_mask] = overlay[i]
    return label


class My_Generator(torch.utils.data.Dataset):

    def __init__(self, thermal_filenames, mask_filenames, batch_size= batch):
        self.thermal_filenames, self.mask_filenames = thermal_filenames, mask_filenames
        self.batch_size = batch_size
        
    def __len__(self):
        return np.uint16(np.ceil(len(self.thermal_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.thermal_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_z = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        thermal = np.zeros((len(batch_x),3,img_row,img_col),dtype=np.float32)      
        masks = np.zeros((len(batch_z),3,img_row,img_col),dtype=np.float32)
            
        for i in range(len(batch_x)): 
            thermal[i] = np.moveaxis(cv2.resize(cv2.imread(batch_x[i]),(img_col,img_row)),-1, 0)        
            masks[i] = np.moveaxis(cv2.resize(cv2.imread(batch_z[i]),(img_col,img_row)),-1, 0)             

        return (thermal/255, masks/255)  
    
class My_val_Generator(torch.utils.data.Dataset):

    def __init__(self, thermal_filenames, mask_filenames, batch_size= batch):
        self.thermal_filenames, self.mask_filenames = thermal_filenames, mask_filenames
        self.batch_size = batch_size
        
    def __len__(self):
        return np.uint16(np.ceil(len(self.thermal_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.thermal_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_z = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        thermal = np.zeros((len(batch_x),3,img_row,img_col),dtype=np.float32)
        masks = np.zeros((len(batch_z),3,img_row,img_col),dtype=np.float32)
            
        for i in range(len(batch_x)): 
            thermal[i] = np.moveaxis(cv2.resize(cv2.imread(batch_x[i]),(img_col,img_row)),-1, 0)    
            masks[i] = np.moveaxis(cv2.resize(cv2.imread(batch_z[i]),(img_col,img_row)),-1, 0)             
        
        return (thermal/255, masks/255)  

def test(args):
    min_loss_t = 1e10 #needs to change for training
    device = args.device
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    my_training_batch_generator = My_Generator(X_train, z_train, batch_size= batch)
    my_validation_batch_generator = My_val_Generator(X_valid, z_valid, batch_size= batch)
    
    ckpt = torch.load("./models/"+args.folder+"/"+args.ckpt)
    model.load_state_dict(ckpt)
    
    k = 0
    for img in X_test:
    
        v_thermal = np.zeros((1,3,img_row,img_col),dtype=np.float32)

        name = img[46:len(img)-4] 
        
        ip = cv2.imread(img)
        
        cv2.imwrite('./results/'+args.folder+'/'+(name) + '_ip.jpg', ip) #save the ip
        x = ip

        ip = cv2.resize(ip, (img_col,img_row))
        
        ip = np.moveaxis(ip, -1, 0)
        
        v_thermal[0] = ip/255

        v_thermal = torch.from_numpy(v_thermal)
        if device=="cuda":
            v_thermal = v_thermal.cuda(non_blocking=True)
            
        for i in tqdm(reversed(range(1, 1000)), position=0):
                t = (torch.ones(1) * i).long().to(args.device)
                predicted_noise = model(v_thermal, t)
                
        op = (predicted_noise[0].detach().cpu().numpy())
        op = torch.ones(op.shape) * (op >= 1)
 
        outp = label_to_mask(op)
    
        cv2.imwrite('./results/'+args.folder+'/'+(name) + '_mask.jpg', outp)
        res = cv2.imread('./results/'+args.folder+'/'+(name) + '_mask.jpg',0)
       
        res = cv2.resize(res, (640,512))
        
        cv2.imwrite('./results/'+args.folder+'/'+(name) + '_mask.jpg', res) #save the mask
        
        img = cv2.imread('./results/'+args.folder+'/'+(name) + '_mask.jpg', 0)
        
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret,y = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #threhold value and the output image
        
        cv2.imwrite('./results/'+args.folder+'/'+(name) + '_finalmask.jpg', y)
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret,y = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #threhold value and the output image
        
        Y = cv2.imread(z_test[k])
        
        cv2.imwrite('./results/'+args.folder+'/'+(name) + '_gt_rgb.jpg', Y) #save the ip
        k+=1
        if((k+1)%100==0):
            print(str(k+1) + " images processed")
        
def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = fname[2:]
    args.epochs = 500
    args.folder = fname[2:]
    args.ckpt = fname[2:] + '_ckpt.pt'
    args.batch_size = 16
    args.image_size = img_size
    args.device = "cuda"
    args.lr = 1e-3
    test(args)


if __name__ == '__main__':
    launch()
