
import Encoder_decoder_disc_VQ as Base_Models
import torch
from torch import nn
import cv2
import numpy as np
from torchvision import models
from torchsummary import summary
import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
#import pandas as pd 
import random 
import torch.utils.data as data_loader
import torchvision
from torchvision import transforms
import torch.nn.functional as F


if torch.cuda.is_available():
    device="cuda" 
    torch.cuda.empty_cache()

else:
    device="cpu"

print("the device is : ",device)


Disc=Base_Models.Disc_net1().to(device)

encode=Base_Models.Encoder().to(device)
VQ_=Base_Models.VQ().to(device)
decode=Base_Models.Decoder().to(device)


print(summary(encode, (3, 128, 128)))

print(summary(decode, ( 32, 12, 12)))
print(summary(Disc, ( 3, 128, 128)))


dataset = torchvision.datasets.ImageFolder(root="test",
                           transform=transforms.Compose([
                               transforms.Resize((128,128)),
                               transforms.ToTensor()
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=True, num_workers=2)
#dataloader=data_loading("C")
criterion = nn.BCELoss()
rec_loss = nn.MSELoss()



lr=0.0001
lr=0.001
params_to_optimize = [{'params': VQ_.parameters()},
    {'params': encode.parameters()},
    {'params': decode.parameters()}
]


optimizerGen = torch.optim.Adam(params_to_optimize, lr=0.001)

#optimizerGen = torch.optim.Adam(params_to_optimize, lr=lr)
optimizerdis = torch.optim.Adam(Disc.parameters(), lr=0.0001)
real_label = 1.
fake_label = 0.


def show_image1(img):
    npimg = img.numpy()
    npimg=(np.transpose(npimg, (1, 2, 0)))*255
    npimg = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test10.jpg",npimg)
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
def train_epoch(encode, decode, device, dataloader, loss_fn,rec_loss_fun, gen_opt,disc_opt):
    # Set train mode for both the encoder and the decoder
    encode.train()
    decode.train()
    train_loss = []
    D_losses = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        disc_opt.zero_grad()
        
        b_size = image_batch.size(0)
        label = torch.ones(b_size, 1).to(device)
        
        #print(image_batch.shape)
        output = Disc(image_batch)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        #errD_real.backward()
        #D_x = output.mean().item()
        
        
        encoded_data = encode(image_batch)
        qloss,quantized,perplexity,encodings=VQ_(encoded_data)
        

        # Decode data
        decoded_data = decode(quantized)
        label = torch.zeros(b_size, 1).to(device)
        output = Disc(decoded_data)
        # Calculate loss on all-real batch
        errD_fake = criterion(output, label)
        # Calculate gradients for D in backward pass
        
        discriminator_loss=errD_fake+errD_real
        
        discriminator_loss.backward()
        disc_opt.step()



        #generator
        
        # Encode data
        encoded_data = encode(image_batch)
        qloss,quantized,perplexity,encodings=VQ_(encoded_data)
        print(qloss)
        b_size = image_batch.size(0)
        label = torch.ones(b_size, 1).to(device)
        

        # Decode data
        decoded_data = decode(quantized)
        loss_rec = rec_loss_fun(decoded_data, image_batch)
        # Evaluate loss
        output = Disc(decoded_data)


        loss2 = loss_fn(output, label)
        loss=loss2+qloss+loss_rec
        # Backward pass
        gen_opt.zero_grad()
        loss.backward()
        gen_opt.step()


        img_recon = decoded_data.cpu()
        image_batch = image_batch.cpu()
        img_recon=torch.cat((img_recon, image_batch), 0)
        codebook = encodings.weight.cpu()
        show_image1(torchvision.utils.make_grid(img_recon[:],16,20))
        codebook = codebook.detach().numpy()
        cv2.imwrite("code.jpg",(codebook*255))
        
        del img_recon,codebook,encoded_data,decoded_data,quantized,image_batch
        torch.cuda.empty_cache()


        
        # Print batch loss
        #print('\t partial train loss (single batch): %f' % (loss.data))\
        D_losses.append(discriminator_loss.detach().cpu().numpy())
        train_loss.append(loss.detach().cpu().numpy())
    print(encodings.weight)
    #print("discriminator_loss",np.mean(D_losses))
    return np.mean(train_loss),np.mean(D_losses)





num_epochs =400
diz_loss = {'generator_loss':[]}
diz2_loss = {'discr_loss':[]}


import time
for epoch in range(num_epochs):
    
   T=time.time()
   Gen_loss,Disc_loss =train_epoch(encode, decode,device,
   dataloader,criterion,rec_loss,optimizerGen,optimizerdis)
   print('\n EPOCH {}/{} \t Gen loss {} '.format(epoch + 1, num_epochs,Gen_loss))
   print('\n EPOCH {}/{} \t Disc loss {} '.format(epoch + 1, num_epochs,Disc_loss))
   diz_loss['generator_loss'].append(Gen_loss)
   diz2_loss['discr_loss'].append(Disc_loss)
   print("time_of_epoch  : ",time.time()-T)
   


plt.figure(figsize=(10,8))
plt.semilogy(diz_loss['generator_loss'], label='gen_loss')
plt.semilogy(diz2_loss['discr_loss'], label='disc_loss')


plt.xlabel('Epoch')
plt.ylabel('Average Loss')
#plt.grid()
plt.legend()
#plt.title('loss')
plt.show()