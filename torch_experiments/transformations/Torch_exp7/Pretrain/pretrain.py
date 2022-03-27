import Model as md
import data as dta
import torchvision.models as models

import torch
import torch.optim as optim
import torch.nn as nn
from pytorch_metric_learning import losses
from torchvision import transforms as T
from torch.utils.data import DataLoader
from absl import app
from absl import flags
from absl import logging
import os
from tqdm import tqdm
from torchlars import LARS
from torch.cuda import amp
import pandas as pd

FLAGS = flags.FLAGS
### Define Flags ###
flags.DEFINE_float('lr', 0.3, 'Learning Rate')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay to use')
flags.DEFINE_float('momentum', 0.9, 'Momentum for optimizer')
flags.DEFINE_float('temperature', 0.1, 'Temperature parameter for NTXent Loss')
flags.DEFINE_integer('image_size', 112, 'Image Size')
flags.DEFINE_string('csv_name', None, 'Name of the csv file with image names')
flags.DEFINE_string('image_dir', None, 'Directory where the images are')
flags.DEFINE_integer('batch_size', 128, 'Batch Size for training')
flags.DEFINE_integer('epochs', 100, 'Epochs to train for')

### End Flags ###

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many arguments')

    logging.info('Training with DenseNet161 as encoder')

    #Remove classifier layer from official model
    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()
            
        def forward(self, x):
            return x

    encoder = models.densenet161(pretrained=False).cuda()
    encoder.classifier = Identity()
    

    #Model to device
    proj_head = md.Projection_Head(2208).cuda() #2208 represents the output shape from DenseNet161

    #optimizer and loss
    base_optimizer = optim.SGD(list(encoder.parameters())+list(proj_head.parameters()),lr=FLAGS.lr,weight_decay=FLAGS.weight_decay,momentum=FLAGS.momentum)
    optimizer = LARS(base_optimizer, trust_coef=0.001)
    ntxent_loss = losses.NTXentLoss(temperature=FLAGS.temperature)

    #Transformations
    transf = T.Compose([
        T.CenterCrop(520),
        T.Resize(FLAGS.image_size),
        T.RandomVerticalFlip(),
        T.RandomHorizontalFlip(),
        T.RandomApply(
        [T.ColorJitter(brightness=0.4, contrast=(0.6,1.4), saturation=(0.6,1.4), hue=0.1),
        T.RandomResizedCrop(FLAGS.image_size, scale=(0.2, 1.0))], p=0.9),
        T.RandomGrayscale(p=0.2),
        T.RandomApply(
        [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.1))], p=0.5),
        T.ToTensor(),
    ])

    #dataset
    astro_ds = dta.AstroDataset(FLAGS.csv_name, FLAGS.image_dir, transform=transf)
    dataset_astro = DataLoader(astro_ds,batch_size=FLAGS.batch_size, shuffle=True,num_workers=6)

    #scheduler cosineAnnealing like in SGDR: Stochastic Gradient Descent with Warm Restarts
    #https://arxiv.org/abs/1608.03983
    #https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, (astro_ds.__len__() * FLAGS.epochs // FLAGS.batch_size + 1))
    #scaler for automatic mixed precision
    scaler = amp.GradScaler()

    def save_model(encoder, projection_head, epoch_number, optimizer, scheduler):

        torch.save({
            'encoder': encoder.state_dict(),
            'projection_head': projection_head.state_dict(),
            'epoch': epoch_number,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, 'model/model.pt')

    start_epoch=0
    if not os.path.isdir('model'):
        os.mkdir('model')

    else:
        #load checkpoints if avalaible
        if os.path.isfile('model/model.pt'):
            logging.info('Restoring from latest checkpoint:')
            checkpoint = torch.load('model/model.pt')
            encoder.load_state_dict(checkpoint['encoder'])
            proj_head.load_state_dict(checkpoint['projection_head'])
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            logging.info('I did not found any checkpoint in model folder. Starting from scratch')

    #Build list for store history of metrics
    df = pd.DataFrame(columns=['Epoch', 'ContrastiveLoss', 'ContrastiveAccuracy'])
    #build softmax
    softmax = nn.Softmax(dim=1)

    for epoch in range(start_epoch, start_epoch + FLAGS.epochs):
        acc_epoch = 0
        epoch_loss = 0
        #use tqdm
        tqdm_loop = tqdm(enumerate(dataset_astro), total=len(dataset_astro), leave=True)
        encoder.train()
        proj_head.train()
        for batch_idx, data in tqdm_loop:
            data = data.cuda()
            transformed_img1, transformed_img2 = torch.split(data, 3, dim=1)
            transformed_img1, transformed_img2 = transformed_img1.cuda(), transformed_img2.cuda()
            inputs = torch.cat((transformed_img1,transformed_img2),0)
            optimizer.zero_grad()
            with amp.autocast():
                projection = proj_head(encoder(inputs))
                pseudolabels = torch.arange(transformed_img1.size(0)).cuda()
                pseudolabels = torch.cat([pseudolabels, pseudolabels], dim=0)
                loss = ntxent_loss(projection, pseudolabels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

            #calculate contrastive accuraccy acording to the oficial implemntation of SimCLR v2
            with torch.no_grad():
                hiddens = torch.split(projection,[projection.size(0)//2,projection.size(0)//2],dim=0)
                logits = torch.matmul(hiddens[0], torch.transpose(hiddens[1], 0,1))/FLAGS.temperature
                logits = softmax(logits)
                contrastive_acc = torch.argmax(logits,dim=1)
                contrastive_acc = (contrastive_acc == pseudolabels[:projection.size(0)//2]).float().sum()
                acc_epoch += contrastive_acc.item()

            #update progress bar
            tqdm_loop.set_description(f'Epoch [{epoch}/{FLAGS.epochs}]')
            tqdm_loop.set_postfix(loss = loss.item())

        save_model(encoder, proj_head, epoch, optimizer, scheduler)
        acc_epoch /= astro_ds.__len__()
        epoch_loss /= (batch_idx+1)
        d_list = [epoch, epoch_loss, acc_epoch]
        df.loc[len(df), :] = d_list
        print('Epoch: {}, Loss: {}, Contrastive Accuracy: {}'.format(epoch, epoch_loss, acc_epoch*100))

    print('Training completed up to {} epochs'.format(epoch+1))
    df.to_csv('pretrain_{}_Epochs_lr{}.csv'.format(epoch+1, FLAGS.lr), index=False)

if __name__ == '__main__':
    app.run(main)