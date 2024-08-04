import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import dumpman
import cv2

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

X = dumpman.undumper('thermal.pkl') #for training only
y = dumpman.undumper('fused_tight.pkl')
z = dumpman.undumper('mask.pkl')

X_train, X_rem, y_train, y_rem, z_train, z_rem = train_test_split(X, y, z, test_size=0.05, random_state=42)


X_valid, X_test, y_valid, y_test, z_valid, z_test = train_test_split(X_rem, y_rem, z_rem, test_size=0.5, random_state=42)

print(X_train[4200])
print(y_train[4200])
print(z_train[4200])

batch = 64
img_row,img_col = 32, 32
img_size = 32
img_row1,img_col1 = 256, 256
dwf = 8

print("Number of training samples: " + str(len(X_train)))
print("Batch size: " + str(batch))

fname = './4'

def encode_half(filters_in,filters_out,bn=True,drop=True):
    if bn and drop:
        return nn.Sequential(
            nn.Conv2d(filters_in, filters_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(0.1),
            nn.BatchNorm2d(filters_out),            
            nn.Dropout(0.5)
        )
    if (bn==True and drop==False):
        return nn.Sequential(
            nn.Conv2d(filters_in, filters_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(0.1),
            nn.BatchNorm2d(filters_out)
        )
    if (bn==False and drop==True):
        return nn.Sequential(
            nn.Conv2d(filters_in, filters_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(0.1),           
            nn.Dropout(0.5)
        )
    if (bn==False and drop==False):
        return nn.Sequential(
            nn.Conv2d(filters_in, filters_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(0.1)
        )

def encode_same(filters_in,filters_out,bn=True,drop=True):
    if bn and drop:
        return nn.Sequential(
            nn.Conv2d(filters_in, filters_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(0.1),
            nn.BatchNorm2d(filters_out),            
            nn.Dropout(0.5)
        )
    if (bn==True and drop==False):
        return nn.Sequential(
            nn.Conv2d(filters_in, filters_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(0.1),
            nn.BatchNorm2d(filters_out)
        )
    if (bn==False and drop==True):
        return nn.Sequential(
            nn.Conv2d(filters_in, filters_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(0.1),           
            nn.Dropout(0.5)
        )
    if (bn==False and drop==False):
        return nn.Sequential(
            nn.Conv2d(filters_in, filters_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(0.1)
        )

def encode_double(filters_in,filters_out,bn=True,drop=True):
    if bn and drop:
        return nn.Sequential(
            nn.ConvTranspose2d(filters_in, filters_out, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ReLU(0.1),
            nn.BatchNorm2d(filters_out),            
            nn.Dropout(0.5)
        )
    if (bn==True and drop==False):
        return nn.Sequential(
            nn.ConvTranspose2d(filters_in, filters_out, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ReLU(0.1),
            nn.BatchNorm2d(filters_out)
        )
    if (bn==False and drop==True):
        return nn.Sequential(
            nn.ConvTranspose2d(filters_in, filters_out, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ReLU(0.1),           
            nn.Dropout(0.5)
        )
    if (bn==False and drop==False):
        return nn.Sequential(
            nn.ConvTranspose2d(filters_in, filters_out, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ReLU(0.1)
        )
    
class DeepVO(nn.Module):
    def __init__(self, dwf):
        super(DeepVO,self).__init__()
        
        self.up = nn.Upsample(scale_factor=int(img_row1/img_row), mode="bilinear", align_corners=True) #upscale mask
        # CNN  
        self.inp = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False) #thermal+mask
        self.e0 = encode_same(4,dwf*2)#256
        self.e1 = encode_half(dwf*2,dwf*4)#128
        self.e2 = encode_half(dwf*4,dwf*8)#64
        self.e3 = encode_half(dwf*8,dwf*16)#32

        self.g3 = encode_same(dwf*16,dwf*16)#32
        self.g2 = encode_double(dwf*16,dwf*8)#64
        self.g1 = encode_double(dwf*8,dwf*4)#128
        self.g0 = encode_double(dwf*4,dwf*2)#256

        self.op1 = nn.Conv2d(dwf*2, 1, kernel_size=3, stride=1, padding=1)
        
        self.sigmoid = nn.Sigmoid()
        

        self.h1 = encode_half(dwf*16,dwf*32)#16
        self.h2 = encode_half(dwf*32,dwf*64)#8
        self.h3 = encode_half(dwf*64,dwf*128)#4
        self.h4 = encode_half(dwf*128,dwf*256)#2
        self.h5 = encode_half(dwf*256,dwf*512)#1
        
        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(dwf*512,dwf*512)
        self.dense2 = nn.Linear(dwf*512,dwf*256)
        self.dense3 = nn.Linear(dwf*256,16)
        self.f2 = encode_double(((dwf*16)), dwf*8, drop=False)#64
        # self.f2 = tensorflow.keras.layers.concatenate([e2,f2],axis = 3)

        self.f1 = encode_double(((dwf*8)+(dwf*8)), dwf*4, drop=False)#128
        # self.f1 = tensorflow.keras.layers.concatenate([e1,f1],axis = 3) 

        self.f0 = encode_double(((dwf*4)+(dwf*4)), 3, drop=False)#256
        # self.f0 = tensorflow.keras.layers.concatenate([e0,f0,op1],axis = 3) 

        self.op2 = nn.Conv2d(((dwf*2)+3), 3, kernel_size=3, stride=1, padding=1) 
        #sigmoid
        
    def forward(self, x, mask): 
        
        # CNN
        z = self.up(mask)
        # print(z.shape, x.shape)
        x = torch.cat((x,z),1)
        x = self.inp(x)
        
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        
        x = self.g3(e3)
        x = self.g3(x)
        x = self.g3(x)
        
        x = self.f2(x)
        x = torch.cat((e2,x),1)
        x = self.f1(x)
        x = torch.cat((e1,x),1)
        x = self.f0(x)
        # print(x.shape)
        x = torch.cat((e0,x),1)        
        x = self.op2(x)
        op2 = self.sigmoid(x)        
        
        return op2    

    def get_loss(self, thermal, mask, rgb):
        predicted_fused = self.forward(thermal, mask)
        # print(predicted_fused.shape, rgb.shape)
        rgb_loss = LogCoshLoss(predicted_fused, rgb)
        loss = rgb_loss 
        return loss

    def step(self, thermal, mask, rgb):
        
        loss = self.get_loss(thermal, mask, rgb)

        return loss

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

    def __init__(self, thermal_filenames, mask_filenames, rgb_filenames, batch_size= batch):
        self.thermal_filenames, self.mask_filenames, self.rgb_filenames = thermal_filenames, mask_filenames, rgb_filenames
        self.batch_size = batch_size
        
    def __len__(self):
        return np.uint16(np.ceil(len(self.thermal_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.thermal_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.rgb_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_z = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        thermal = np.zeros((len(batch_x),3,img_row,img_col),dtype=np.float32)   
        fused = np.zeros((len(batch_y),3,img_row1,img_col1),dtype=np.float32)   
        masks = np.zeros((len(batch_z),3,img_row,img_col),dtype=np.float32)
        thermal_full = np.zeros((len(batch_x),1,img_row1,img_col1),dtype=np.float32) 
            
        for i in range(len(batch_x)): 
            thermal[i] = np.moveaxis(cv2.resize(cv2.imread(batch_x[i]),(img_col,img_row)),-1, 0)      
            fused[i] = np.moveaxis(cv2.resize(cv2.imread(batch_y[i]),(img_col1,img_row1)),-1, 0)  
            masks[i] = np.moveaxis(cv2.resize(cv2.imread(batch_z[i]),(img_col,img_row)),-1, 0) 
            thermal_full[i] = np.moveaxis(cv2.resize(cv2.imread(batch_x[i],0),(img_col1,img_row1)),-1, 0)

        return (thermal/255, masks/255, fused/255, thermal_full/255)  
    
class My_val_Generator(torch.utils.data.Dataset):

    def __init__(self, thermal_filenames, mask_filenames, rgb_filenames, batch_size= batch):
        self.thermal_filenames, self.mask_filenames, self.rgb_filenames = thermal_filenames, mask_filenames, rgb_filenames
        self.batch_size = batch_size
        
    def __len__(self):
        return np.uint16(np.ceil(len(self.thermal_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.thermal_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.rgb_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_z = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        thermal = np.zeros((len(batch_x),3,img_row,img_col),dtype=np.float32)   
        fused = np.zeros((len(batch_y),3,img_row1,img_col1),dtype=np.float32)   
        masks = np.zeros((len(batch_z),3,img_row,img_col),dtype=np.float32)
        thermal_full = np.zeros((len(batch_x),1,img_row1,img_col1),dtype=np.float32) 
            
        for i in range(len(batch_x)): 
            thermal[i] = np.moveaxis(cv2.resize(cv2.imread(batch_x[i]),(img_col,img_row)),-1, 0)      
            fused[i] = np.moveaxis(cv2.resize(cv2.imread(batch_y[i]),(img_col1,img_row1)),-1, 0)  
            masks[i] = np.moveaxis(cv2.resize(cv2.imread(batch_z[i]),(img_col,img_row)),-1, 0) 
            thermal_full[i] = np.moveaxis(cv2.resize(cv2.imread(batch_x[i],0),(img_col1,img_row1)),-1, 0)

        return (thermal/255, masks/255, fused/255, thermal_full/255)   

def LogCoshLoss(y_t, y_prime_t):
    ey_t = y_t - y_prime_t
    return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

def train(args):
    min_loss_t = 1e10 #needs to change for training
    device = args.device
    model = UNet().to(device)
    model1 = DeepVO(dwf).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    # loss_fn = LogCoshLoss
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    # l = len(dataloader)
    my_training_batch_generator = My_Generator(X_train, z_train, y_train, batch_size= batch)
    my_validation_batch_generator = My_val_Generator(X_valid, z_valid, y_valid, batch_size= batch)
    
    ckpt = torch.load("./models/DDPM_Unconditional/206_ckpt.pt")
    model.load_state_dict(ckpt)
    
    print("Number of iterations per epoch = " + str(len(my_training_batch_generator)))
    
    patience = 0

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch+1}:")
        # pbar = tqdm(dataloader)
        loss_mean1, loss_mean2 = 0, 0
        i = 0
        
        for t_thermal, t_y, t_fused, t_full in my_training_batch_generator:
            t_y = torch.from_numpy(t_y)
            t_thermal = torch.from_numpy(t_thermal)
            t_fused = torch.from_numpy(t_fused)
            t_full = torch.from_numpy(t_full)
            if device=="cuda":            
                t_y = t_y.cuda(non_blocking=True)
                t_thermal = t_thermal.cuda(non_blocking=True)
                t_fused = t_fused.cuda(non_blocking=True)
                t_full = t_full.cuda(non_blocking=True)
            t = diffusion.sample_timesteps(t_thermal.shape[0]).to(device)
            x_t = diffusion.noise_images(t_thermal, t)
            predicted_noise = model(x_t, t)
            loss1 = loss_fn(t_y, predicted_noise)
            
            if(i == len(my_training_batch_generator)):
                print("Ending epoch")
                break
                
            loss_mean1 += float(loss1)       
     
                
            if(np.isnan(loss_mean1) or np.isnan(loss_mean2)):
                print("NaN during iterations")
                print(t_thermal, t_y)
                break
                
            loss2 = model1.get_loss(t_full, predicted_noise, t_fused)
            loss_mean2 += float(loss2)
            
            if((i+1)%100==0):
                print("loss1 at step " + str(i+1) + " is " + str(loss_mean1/(i+1)))
                print("loss2 at step " + str(i+1) + " is " + str(loss_mean2/(i+1)))
            
            loss = loss1 + loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i+=1

        loss_mean1 /= len(X_train)
        loss_mean2 /= len(X_train)
        f = open(fname, 'a')
        f.write('Epoch {}\ntrain loss1 mean: {} '.format(epoch+1, loss_mean1))
        f.write('train loss2 mean: {}\n'.format(loss_mean2))
        print('Epoch {}\ntrain loss1 mean: {} '.format(epoch+1, loss_mean1))
        print('train loss2 mean: {}\n'.format(loss_mean2))

        # Save model

        # save if the training loss decrease
        check_interval = 1
        if loss_mean1 < min_loss_t and epoch % check_interval == 0:
            min_loss_t = loss_mean1
            print('Save model at ep {}, mean of train loss: {}'.format(epoch+1, loss_mean1 + loss_mean2))
            torch.save(model.state_dict(), fname+'_model.train')
            torch.save(model1.state_dict(), fname+'_DeepVO.train')
            torch.save(optimizer.state_dict(), fname+'_opt.train')
            patience = 0
            print("Learning rate: " + str(args.lr))
            
        else:
            patience+=1
            if(patience>=3):
                args.lr/=10
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                patience = 0
            print("Learning rate: " + str(args.lr))
            
        f.close()
           
        for t_thermal, t_y, t_fused, t_full in my_validation_batch_generator:    
            sampled_images = diffusion.sample(model, t_thermal, n=t_thermal.shape[0])
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"{epoch}_ckpt.pt"))
            torch.save(model1.state_dict(), os.path.join("models", args.run_name, f"{epoch}_DeepVO_ckpt.pt"))
            break


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Unconditional_4"
    args.epochs = 500
    args.batch_size = 16
    args.image_size = img_size
    args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
    args.device = "cuda"
    args.lr = 1e-3
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
