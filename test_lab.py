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
z = dumpman.undumper('fused_tight.pkl')

X_train, X_rem, z_train, z_rem = train_test_split(X, z, test_size=0.05, random_state=42)


X_valid, X_test, z_valid, z_test = train_test_split(X_rem, z_rem, test_size=0.5, random_state=42)

batch = 64
img_row,img_col = 32, 32
img_size = 32

print("Number of testing samples: " + str(len(X_test)))

fname = './fused_3'

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = x
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, x, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(x)
            x = x.to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        x = (predicted_noise * 255).type(torch.uint8)
        return x
    
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

def LogCoshLoss(y_t, y_prime_t): 
    ey_t = y_t - y_prime_t
    return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

def test(args):
    min_loss_t = 1e10 #needs to change for training
    device = args.device
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    # loss_fn = LogCoshLoss
    diffusion = Diffusion(img_size=args.image_size, device=device)
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
                
        op = (predicted_noise[0].detach().cpu().numpy())*255
        cv2.imwrite('./results/'+args.folder+'/'+(name) + '_mask.jpg', op[0])
        res = cv2.imread('./results/'+args.folder+'/'+(name) + '_mask.jpg',0)
       
        res = cv2.resize(res, (640,512))
        
        cv2.imwrite('./results/'+args.folder+'/'+(name) + '_mask.jpg', res) #save the mask
        
        img = cv2.imread('./results/'+args.folder+'/'+(name) + '_mask.jpg', 0)
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret,y = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #threhold value and the output image
    
        cv2.imwrite('./results/'+args.folder+'/'+(name) + '_finalmask.jpg', y)
        l = ((y/255) * x[:,:,0])
        cv2.imwrite('./results/'+args.folder+'/'+(name) + '_fused.jpg', l)
        a = cv2.resize(op[1], (640,512)) - 128
        b = cv2.resize(op[2], (640,512)) - 128
        
        lab = np.zeros((l.shape[0],l.shape[1],3))
        lab[:,:,0] = (l/255)*100
        lab[:,:,1] = a
        lab[:,:,2] = b
        
        rgb = lab2rgb(lab)
        
        cv2.imwrite('./results/'+args.folder+'/'+(name) + '_op.jpg', rgb*255)
        Y = cv2.imread(z_test[k], 0)
        
        cv2.imwrite('./results/'+args.folder+'/'+(name) + '_gt.jpg', Y) #save the ip
        k+=1
        if((k+1)%100==0):
            print(str(k+1) + " images processed")
        
def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Unconditional_fused_3"
    args.epochs = 500
    args.folder = 'DDPM_Unconditional_fused_3'
    args.ckpt = 'fused_3_ckpt.pt'
    args.batch_size = 16
    args.image_size = img_size
    args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
    args.device = "cuda"
    args.lr = 1e-3
    test(args)


if __name__ == '__main__':
    launch()
