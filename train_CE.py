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
import time
from skimage import color, io 
import torch.nn.functional as F

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

X = dumpman.undumper('thermal.pkl') #for training only
y = dumpman.undumper('mask.pkl')

X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.05, random_state=42)


X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

print(X_train[4700], y_train[4700])

batch = 64
img_row,img_col = 32, 32
img_size = 32

print("Number of training samples: " + str(len(X_train)))
print("Batch size: " + str(batch))

fname = 'fused_3_CE'

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
        batch_y = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        thermal = np.zeros((len(batch_x),3,img_row,img_col),dtype=np.float32)      
        masks = np.zeros((len(batch_y),1,img_row,img_col),dtype=np.float32)
            
        for i in range(len(batch_x)): 
            thermal[i] = np.moveaxis(cv2.resize(cv2.imread(batch_x[i]),(img_col,img_row)),-1, 0)   
            masks[i] = (cv2.resize(cv2.imread(batch_y[i],0),(img_col,img_row)))

        return (thermal/255, masks/255)  
    
class My_val_Generator(torch.utils.data.Dataset):

    def __init__(self, thermal_filenames, mask_filenames, batch_size= batch):
        self.thermal_filenames, self.mask_filenames = thermal_filenames, mask_filenames
        self.batch_size = batch_size
        
    def __len__(self):
        return np.uint16(np.ceil(len(self.thermal_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.thermal_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        thermal = np.zeros((len(batch_x),3,img_row,img_col),dtype=np.float32)      
        masks = np.zeros((len(batch_y),1,img_row,img_col),dtype=np.float32)
            
        for i in range(len(batch_x)): 
            thermal[i] = np.moveaxis(cv2.resize(cv2.imread(batch_x[i]),(img_col,img_row)),-1, 0)    
            masks[i] = (cv2.resize(cv2.imread(batch_y[i],0),(img_col,img_row)))

        return (thermal/255, masks/255) 

class_index = [0,1]
color_code = [0,255]

def label_to_mask(mask):
    label = np.zeros((mask.shape[1],mask.shape[2]), dtype=np.uint8)
    
    for i, class_val in enumerate(class_index):
            bool_mask = np.array(
                (mask == 1),
                dtype=bool,
            )           
            label = color_code[i]
    return label
    
def LogCoshLoss(y_t, y_prime_t): # An interesting loss to check for: not implemented here
    ey_t = y_t - y_prime_t
    return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
   
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    mask = target >= 0
    target = target[mask]
    target = target.type(torch.LongTensor)
    target = target.cuda()
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

def train(args):
    min_loss_t = 1e10 #needs to change for training
    device = args.device
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # loss_fn = nn.MSELoss() # this should be used for RGB
    loss_fn = cross_entropy2d
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    my_training_batch_generator = My_Generator(X_train, y_train, batch_size= batch)
    my_validation_batch_generator = My_val_Generator(X_valid, y_valid, batch_size= batch)
    
    ckpt = torch.load("./models/fused_3_CE/fused_3_CE_ckpt.pt")
    model.load_state_dict(ckpt)
    
    print("Number of iterations per epoch = " + str(len(my_training_batch_generator)))
    
    patience = 0

    for epoch in range(args.epochs):
        tim = time.time()
        logging.info(f"Starting epoch {epoch+1}:")
        loss_mean = 0
        i = 0
        
        for t_thermal, t_y in my_training_batch_generator:
            t_y = torch.from_numpy(t_y)
            t_thermal = torch.from_numpy(t_thermal)
            if device=="cuda":            
                t_y = t_y.cuda(non_blocking=True)
                t_thermal = t_thermal.cuda(non_blocking=True)
            t = diffusion.sample_timesteps(t_thermal.shape[0]).to(device)
            x_t = diffusion.noise_images(t_thermal, t)
            predicted_noise = model(x_t, t)
            loss = loss_fn(predicted_noise, t_y)
            
            if(i == len(my_training_batch_generator)):
                print("Ending epoch")
                break
                
            loss_mean += float(loss)
            
            if((i+1)%100==0):
                print("The loss at step " + str(i+1) + " is " + str(loss_mean/(i+1)))
                
            if(np.isnan(loss_mean)):
                print("NaN during iterations")
                print(t_thermal, t_y)
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i+=1

        loss_mean /= len(X_train)
        print('Time taken: {}\n'.format(tim - time.time()))
        f = open(fname, 'a')
        f.write('Epoch {}\ntrain loss mean: {} Learning Rate: {}\n'.format(epoch+1, loss_mean, args.lr))
        print('Epoch {}\ntrain loss mean: {}\n'.format(epoch+1, loss_mean))

        # Save model

        # save if the training loss decrease
        check_interval = 1
        if loss_mean < min_loss_t and epoch % check_interval == 0:
            min_loss_t = loss_mean
            print('Save model at ep {}, mean of train loss: {}'.format(epoch+1, loss_mean))
            torch.save(model.state_dict(), fname+'_model.train')
            torch.save(optimizer.state_dict(), fname+'_opt.train')
            patience = 0
            print("Learning rate: " + str(args.lr))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"{fname}_ckpt.pt"))
            
        else:
            patience+=1
            if(patience>=3):
                args.lr/=10
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                patience = 0
            print("Learning rate: " + str(args.lr))
            
        f.close()
           
        for t_thermal, t_y in my_validation_batch_generator:    
            sampled_images = diffusion.sample(model, t_thermal, n=t_thermal.shape[0])
            x = label_to_mask(sampled_images)
            save_images(x, os.path.join("results", args.run_name, f"{epoch+1}.jpg"))
            break


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "fused_3_CE"
    args.epochs = 450
    args.batch_size = 16
    args.image_size = img_size
    args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
    args.device = "cuda"
    args.lr = 1e-3
    train(args)


if __name__ == '__main__':
    launch()
