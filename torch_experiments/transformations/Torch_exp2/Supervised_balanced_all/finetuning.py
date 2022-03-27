import Model as md
import Resnet as res
import data as dta

import torch
import torch.nn as nn
import torch.optim as optim
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

flags.DEFINE_integer('resnet_depth', 50, 'Resnet Depth')
flags.DEFINE_float('lr', 0.01, 'Learning Rate')
flags.DEFINE_float('weight_decay', 0, 'Weight decay to use')
flags.DEFINE_float('momentum', 0.9, 'Momentum for optimizer')
flags.DEFINE_integer('image_size', 112, 'Image Size')
flags.DEFINE_string('csv_name', None, 'Name of the csv file with image names for finetuning')
flags.DEFINE_string('csv_name_val', None, 'Name of the csv file with image names for validation')
flags.DEFINE_string('image_dir', None, 'Directory where the images for finetuning are')
flags.DEFINE_string('image_dir_val', None, 'Directory where the images for validation are')
flags.DEFINE_integer('batch_size', 128, 'Batch Size for training')
flags.DEFINE_string('checkpoint_dir', None, 'Model directory where checkpoint from pretrain is')
flags.DEFINE_integer('epochs', 90, 'Epochs to train for')
flags.DEFINE_bool('start_from_scratch', False, 'Determine if start taining from scratch (supervised) or not (semisupervised)')
flags.DEFINE_integer('projection_head_selector', 1, 'which linear layer to use, if 0 use the output from Resnet')
flags.DEFINE_integer('number_of_classes', 5, 'How many classes in dataset are')
### End Flags ###

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many arguments')

    #Check coherence in projection_head_selector
    if FLAGS.projection_head_selector < 0 or FLAGS.projection_head_selector>3:
        raise ValueError("I was constructed with 3 dense layers in projection head. projection_head_selector must be between 0 and 3")

    #Build base enconder:
    if FLAGS.resnet_depth == 50:
        encoder = res.ResNet50().cuda()
    elif FLAGS.resnet_depth == 101:
        encoder = res.ResNet101().cuda()
    elif FLAGS.resnet_depth == 152:
        encoder =  res.ResNet152().cuda()
    else:
        raise ValueError("Not Implemented Resnet :c")
    
    #Build proyection head
    if FLAGS.projection_head_selector == 0:
        proj_head = None
    else:
        proj_head = md.Projection_Head(encoder.representation_dim, FLAGS.projection_head_selector).cuda()

    #Build supervised head
    sup_head = md.Supervised_Head(proj_head.output if proj_head is not None else encoder.representation_dim, FLAGS.number_of_classes).cuda()

    #Build optimizer and supervised loss
    if proj_head is not None:
        base_optimizer = optim.SGD(list(encoder.parameters())+list(proj_head.parameters())+list(sup_head.parameters()),lr=FLAGS.lr,weight_decay=FLAGS.weight_decay,momentum=FLAGS.momentum)
    else:
        base_optimizer = optim.SGD(list(encoder.parameters())+list(sup_head.parameters()),lr=FLAGS.lr,weight_decay=FLAGS.weight_decay,momentum=FLAGS.momentum)
    optimizer = LARS(base_optimizer, trust_coef=0.001)
    cross_entropy_loss = nn.CrossEntropyLoss()

    #Define Transformations
    #Transformations for finetuning
    transf = T.Compose([
        T.CenterCrop(520),
        T.Resize(FLAGS.image_size),
        T.RandomVerticalFlip(),
        T.RandomHorizontalFlip(),
        T.RandomApply(
        [T.RandomResizedCrop(FLAGS.image_size, scale=(0.2, 1.0))], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
    ])

    #Transformations for validation
    transf_val = T.Compose([
        T.CenterCrop(412),
        T.Resize(FLAGS.image_size),
        T.ToTensor(),
    ])

    #Build dataset and data loader for finetuning
    astro_ds = dta.AstroDataset(FLAGS.csv_name, FLAGS.image_dir, transform=transf)
    dataset_astro = DataLoader(astro_ds,batch_size=FLAGS.batch_size, shuffle=True,num_workers=6)
    #Build dataset and data loader for validation
    astro_ds_val = dta.AstroDataset(FLAGS.csv_name_val, FLAGS.image_dir_val, transform=transf_val)
    dataset_astro_val = DataLoader(astro_ds_val,batch_size=FLAGS.batch_size, shuffle=False,num_workers=6)

    #scheduler cosineAnnealing like in SGDR: Stochastic Gradient Descent with Warm Restarts
    #https://arxiv.org/abs/1608.03983
    #https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, (astro_ds.__len__() * FLAGS.epochs // FLAGS.batch_size + 1))
    #scaler for automatic mixed precision
    scaler = amp.GradScaler()

    def save_model(encoder, projection_head, sup_head, epoch_number, optimizer, scheduler):
        
        if projection_head is not None:
            torch.save({
                'encoder': encoder.state_dict(),
                'projection_head': projection_head.state_dict(),
                'supervised_head': sup_head.state_dict(),
                'epoch': epoch_number,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, 'finetuned_model/model.pt')

        else:
            torch.save({
                'encoder': encoder.state_dict(),
                'supervised_head': sup_head.state_dict(),
                'epoch': epoch_number,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, 'finetuned_model/model.pt')

    start_epoch=0
    if not os.path.isdir('finetuned_model'):
        os.mkdir('finetuned_model')

    # else:
    #     #load checkpoints if avalaible
    #     if os.path.isfile('finetuned_model/model.pt'):
    #         logging.info('Restoring from latest best checkpoint:')
    #         checkpoint = torch.load('finetuned_model/model.pt')
    #         encoder.load_state_dict(checkpoint['encoder'])
    #         if proj_head is not None:
    #             proj_head.load_state_dict(checkpoint['projection_head'], strict=False)
    #         start_epoch = checkpoint['epoch']
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         scheduler.load_state_dict(checkpoint['scheduler'])
    #     else:
    #         logging.info('I did not found any checkpoint in finetuned_model folder. Starting finetuning from scratch')

    #Build list for store history of metrics
    df_train = pd.DataFrame(columns=['Epoch','SupervisedLoss','Accuracy'])
    df_eval = pd.DataFrame(columns=['Epoch','Accuracy'])

    #Load checkpoints from pretrain
    if not FLAGS.start_from_scratch:
        logging.info('Loading weights from checkpoint in checkpoint_dir:{}'.format(FLAGS.checkpoint_dir))
        if os.path.isfile(FLAGS.checkpoint_dir + '/model.pt'):
            checkpoint = torch.load(FLAGS.checkpoint_dir + '/model.pt')
            encoder.load_state_dict(checkpoint['encoder'])
            if proj_head is not None:
                proj_head.load_state_dict(checkpoint['projection_head'], strict=False)
            logging.info('Successfully loaded model.pt in {}'.format(FLAGS.checkpoint_dir))
        else:
            raise ValueError('I did not found model.pt file in folder {}'.format(FLAGS.checkpoint_dir))

    #build softmax
    softmax = nn.Softmax(dim=1)
    #set constant value for best accuracy in validation
    best_acc_till_now = 0
    best_epoch = 0

    #Start training loop
    for epoch in range(start_epoch, start_epoch + FLAGS.epochs):
        acc_epoch = 0
        epoch_loss = 0
        #use tqdm for finetuning
        tqdm_loop = tqdm(enumerate(dataset_astro), total=len(dataset_astro), leave=True)
        encoder.train()
        if proj_head is not None:
            proj_head.train()
        sup_head.train()

        for batch_idx, data in tqdm_loop:
            image = data[0].cuda()
            labels = data[1].cuda()
            optimizer.zero_grad()

            with amp.autocast():
                output = sup_head(proj_head(encoder(image))) if proj_head is not None else sup_head(encoder(image))
                loss = cross_entropy_loss(output, labels)
                epoch_loss += loss.item()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            #update progress bar
            tqdm_loop.set_description(f'Epoch [{epoch}/{FLAGS.epochs}]')
            tqdm_loop.set_postfix(loss = loss.item())

            with torch.no_grad():
                output = softmax(output)
                acc = torch.argmax(output, dim=1)
                acc = (acc == labels).float().sum()
                acc_epoch += acc
        
        acc_epoch /= astro_ds.__len__()
        epoch_loss /= (batch_idx+1)
        d_list = [epoch, epoch_loss, acc_epoch.item()]
        df_train.loc[len(df_train), :] = d_list
        logging.info('Epoch: {}, Loss: {}, Supervised Accuracy: {}'.format(epoch, epoch_loss, acc_epoch*100))

        #set model to eval mode to evaluate with validation data
        encoder.eval()
        if proj_head is not None:
            proj_head.eval()
        sup_head.eval()
        acc_val = 0
        with torch.no_grad():
            for data_val in dataset_astro_val:
                image_val = data_val[0].cuda()
                label_val = data_val[1].cuda()

                #only forward pass
                with amp.autocast():
                    output_val = sup_head(proj_head(encoder(image_val))) if proj_head is not None else sup_head(encoder(image_val))
                    output_val = softmax(output_val)
                    acc_valid = torch.argmax(output_val,dim=1)
                    acc_valid = (acc_valid == label_val).float().sum()
                    acc_val += acc_valid
                
            acc_val /= astro_ds_val.__len__()
        eval_list = [epoch, acc_val.item()]
        df_eval.loc[len(df_eval), :] = eval_list
        logging.info('Validation step at Epoch {}, Supervised Accuracy: {}'.format(epoch, acc_val*100))

        #save model
        if acc_val.item() > best_acc_till_now:
            best_acc_till_now = acc_val.item()
            best_epoch = epoch
            save_model(encoder, proj_head, sup_head, epoch, optimizer, scheduler)

    logging.info('Finetuning completed up to {} epocs. Best Epoch was {}, with acc {}'.format(epoch+1, best_epoch, best_acc_till_now))
    df_train.to_csv('Finetuning_{}_Epochs_lr{}_train.csv'.format(epoch+1, FLAGS.lr), index=False)
    df_eval.to_csv('Finetuning_{}_Epochs_lr{}_valid.csv'.format(epoch+1, FLAGS.lr), index=False)

if __name__ == '__main__':
    app.run(main)