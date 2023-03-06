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
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn10 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool1(x)
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = nn.functional.relu(self.bn5(self.conv5(x)))
        x = nn.functional.relu(self.bn6(self.conv6(x)))
        x = self.pool2(x)
        x = nn.functional.relu(self.bn7(self.conv7(x)))
        x = nn.functional.relu(self.bn8(self.conv8(x)))
        x = nn.functional.relu(self.bn9(self.conv9(x)))
        x = self.pool3(x)
        x = self.bn10(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = nn.functional.relu(self.bn5(self.conv5(x)))
        x = nn.functional.relu(self.bn6(self.conv6(x)))
        x = nn.functional.sigmoid(self.conv7(x))
        return x



class VQ (nn.Module ):
    def __init__(self) :
        super().__init__()
        self.word_embedding_dim = 32 #if we make it 32 then change above in encoder and decoder
        self._num_embeddings = 64
        
        self._embedding = nn.Embedding(self._num_embeddings, self.word_embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        
        self._commitment_cost = 0.25

        
    def forward(self,x):
        inputs=x.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        x=inputs.view(-1,self.word_embedding_dim)

        distances = (torch.sum(x**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(x, self._embedding.weight.t()))
        encoding_indices=torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        #encoding_indices=torch.nn.functional.one_hot(encoding_indices,self._num_embeddings)
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        #perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        perplexity=1
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, self._embedding


import torch.nn as nn

class Disc_net1(nn.Module):
    def __init__(self):
        super(Disc_net1, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=64*16*16, out_features=256)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(in_features=256, out_features=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = x.view(-1, 64*16*16)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

Disc=Disc_net1().to(device)

encode=Encoder().to(device)
VQ_=VQ().to(device)
decode=Decoder().to(device)
#print(model)
print(summary(encode, (3, 128, 128)))

print(summary(decode, ( 32, 12, 12)))
print(summary(Disc, ( 3, 128, 128)))

# def init_all(model, init_func, *params, **kwargs):
#     for p in model.parameters():
#         init_func(p, *params, **kwargs)

# init_all(encode, torch.nn.init.normal_, mean=0.1, std=0.7)
# init_all(decode, torch.nn.init.normal_, mean=0.1, std=0.7)
# # init_all(VQ_, torch.nn.init.normal_, mean=0., std=1)
# init_all(Disc, torch.nn.init.normal_, mean=0.1, std=0.7)




# def data_loading(_dir):
#     tr=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((128,128))
#     ])
    
#     data_images=torchvision.datasets.ImageFolder(root=str(_dir),transform=transforms.Compose([
#         transforms.ToTensor(),transforms.Resize((128,128))]))
#     #data_images.transform = tr
#     dataloader = torch.utils.data.DataLoader(data_images, batch_size=64,
#                                          shuffle=True,num_workers=2)
    # for X, y in train_dataloader:
    #     print(f"Shape of X [N, C, H, W]: {X.shape}")
    #     print(X[0])
    #     print(f"Shape of y: {y.shape} {y.dtype}")
    #     break
    # return dataloader
dataset = torchvision.datasets.ImageFolder(root="102flowers",
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
# params_to_optimize = [
#     {'params': VQ_.parameters()},
#     {'params': encode.parameters()},
#     {'params': decode.parameters()}
# ]

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
    print("discriminator_loss",np.mean(D_losses))
    return np.mean(train_loss)



import cv2

def show_image(epoch,img):
    npimg = img.numpy()
    #print(npimg.shape)
    cv2.imwrite(str(epoch)+"test.jpg",(np.transpose(npimg, (1, 2, 0))*255))
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
#

def show_me(epoch):
    encode.eval()
    decode.eval()
    with torch.no_grad():
        # calculate mean and std of latent code, generated takining in test images as inputs 
        # for X, y in dataloader:
        #     print(f"Shape of X [N, C, H, W]: {X.shape}")
        #     #print(X[0])
        #     print(f"Shape of y: {y.shape} {y.dtype}")
        #     break
        # #images, labels = iter(data_loader).next()
        # images = X.to(device)
        # latent = encode(images)
        # latent = latent.cpu()

        # mean = latent.mean(dim=0)
        # print(mean)
        # std = (latent - mean).pow(2).mean(dim=0).sqrt()
        # print(std)

        # sample latent vectors from the normal distribution
        #torch.manual_seed(0)
        latent = (torch.randn(110, 32, 8, 8)-0.5)/0.5
        #latent = torch.randn(110, 32, 8, 8)

        # reconstruct images from the random latent vectors
        latent = latent.to(device)
        img_recon = decode(latent)
        qloss,latent,perplexity,encodings=VQ_(latent)
        img_recon = decode(latent)
        img_recon = img_recon.cpu()
        encodings = encodings.weight.cpu()
        print(encodings.shape)
        fig, ax = plt.subplots(figsize=(20, 8.5))
        show_image(epoch,torchvision.utils.make_grid(img_recon[:100],5,10))
        #show_image(str(epoch)+"a",encodings)
        
        #plt.show()
        #print("done")
        


num_epochs = 400
diz_loss = {'train_loss':[]}


import time
for epoch in range(num_epochs):
    
   T=time.time()
   #show_me(epoch)
   train_loss =train_epoch(encode, decode,device,
   dataloader,criterion,rec_loss,optimizerGen,optimizerdis)
   #val_loss = test_epoch(encoder,decoder,device,test_loader,loss_fn)
   print('\n EPOCH {}/{} \t train loss {} '.format(epoch + 1, num_epochs,train_loss))
   diz_loss['train_loss'].append(train_loss)
   print("time_of_epoch  : ",time.time()-T)
   #scha1.step()
   
#    if (epoch+1)%5==0:
#     show_me()

   #diz_loss['val_loss'].append(val_loss)
   #plot_ae_outputs(encode,decode,n=10)


#show_me(epoch)

plt.figure(figsize=(10,8))
plt.semilogy(diz_loss['train_loss'], label='Train')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
#plt.grid()
plt.legend()
#plt.title('loss')
plt.show()