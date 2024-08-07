# pedestrian-masks
Code for An Image Information Fusion based Simple Diffusion Network leveraging the Segment Anything Model for Guided Attention on Thermal Images producing Colorized Pedestrian Masks

This code is for the paper at https://www.sciencedirect.com/science/article/abs/pii/S1566253524003968 
The code is for training a model to work with the KAIST Multispectral Pedestrian dataset, to use diffusion for creating a tight pedestrian mask and then creating a RGB-IR fused mask for pedestrian detection.
https://soonminhwang.github.io/rgbt-ped-detection/

The code is based on https://github.com/tcapelle/Diffusion-Models-pytorch

There are 2 set of codes available for training: 
1. train_rgb
2. train_CE
   
train_rgb works on RGB images for both input (thermal is taken as a 3 channel input image to make use of the diffusion module which uses 3 channel input) and the output, while train_CE is based on 2D Cross Entropy loss (it has 2 channels, where 1 is the pedestrian class).

The data cards are provided as:
1. mask.pkl
2. rgb.pkl
3. thermal.pkl

These are the binarized data cards, for reproucibility. Other than this, the train test and validation for the data is also available at the IEEE DataPort link, where the images for the mask files can be obtained.
Please note that the mask files are created by employing the Segment Anything Model (SAM: https://segment-anything.com/) on the annotations provided by the authors for thermal images in the KAIST Multispectral Pedestrian Dataset. We use only 1 object in our use case for the dataset, but more research can be conducted on the multi-class classification for the same.

Link for checkpoints: 
1. train_rgb: https://drive.google.com/file/d/10S5oQ7K4Tx3TeZi817Ix2OIzFILDi_fC/view?usp=sharing
2. train_CE: https://drive.google.com/file/d/1V6pevnY_v5FGTBoswIwHDHmkdcFL_aX5/view?usp=sharing

<img width="765" alt="Screenshot 2024-08-04 at 7 43 43 PM" src="https://github.com/user-attachments/assets/8149fd22-7ba7-4e7f-b5eb-564c7aca8cec">
