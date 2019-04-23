# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:24:20 2019

@author: yao
"""
import os
from torchvision import transforms 
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torch
import argparse
import time
from torch import optim
import numpy as np
#from deeplabv2_resnet101 import Deeplabv2
from DeepLabv3 import DeepLabV3
from DeepLabv3Plus import DeepLabv3_plus
from res_unet import ResUnet
from FCN import FCN8
from torch.autograd import Variable

import matplotlib.pyplot as plt
# =============================================================================
# 设置超参数
# =============================================================================
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#export CUDA_VISIBLE_DEVICES=0
parse=argparse.ArgumentParser()
parse.add_argument('--batch_size',type=int,default=4)
parse.add_argument('--model_path',type=str,default='./model/')
parse.add_argument('--save_valid_path',type=str,default='./valid/Benign/')
#parse.add_argument('--save_valid_path',type=str,default='./valid/Maligant/')
#parse.add_argument('--model_name',type=str,default='deeplabv3_M')
parse.add_argument('--model_name',type=str,default='deeplabv3_plus_1')
#parse.add_argument('--model_name',type=str,default='Unet_M')
#parse.add_argument('--model_name',type=str,default='FCN_M')

parse.add_argument('--train',type=bool,default=False)
parse.add_argument('--test',type=bool,default=True)
parse.add_argument('--validation',type=bool,default=False)
parse.add_argument('--epochs',type=int,default=80)
parse.add_argument('--lr',type=float,default=0.0001)
parse.add_argument('--display_epoch',type=int,default=5)
parse.add_argument('--num_class',type=int,default=2)
parse.add_argument('--out-stride', type=int, default=8,
                   help='network output stride (default: 8)')
parse.add_argument('--pretrained',type=bool,default=False)

args=parse.parse_args()
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)
action=None
# =============================================================================
# 加载数据
# =============================================================================
transform1=transforms.Compose([
        transforms.Resize(512),
#        transforms.CenterCrop(224),
        transforms.FiveCrop(224),
#        transforms.TenCrop(size, vertical_flip=False)
#        transforms.ToTensor()
        transforms.Lambda(lambda crops:torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ])
transform2=transforms.Compose([
        transforms.Resize((256,256)),
#        transforms.FiveCrop(224),
#        transforms.Lambda(lambda crops:torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        transforms.ToTensor()
])
transform3=transforms.Compose([
        transforms.ToTensor()
])
class create_dataset(Dataset):
    def __init__(self,root):
        self.root=os.listdir(root)
        if action=='train':
#        if args.train:
            self.image_transform=transform1
            self.label_transform=transform1
        if action=='test':
#        if args.test:
            self.image_transform=transform2
            self.label_transform=transform2
        if action=='validation':
            self.image_transform=transform3
            self.label_transform=transform3
        self.image_dir=os.path.join(root,'B_Images')
        self.label_dir=os.path.join(root,'B_Labels')
        self.image_list=[]
        self.label_list=[]
        for image in os.listdir(self.image_dir):
            self.image_list.append(image)
        for label in os.listdir(self.label_dir):
            self.label_list.append(label)
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,index):
        image_name=os.path.join(self.image_dir,self.image_list[index])
        label_name=os.path.join(self.label_dir,self.label_list[index])
        image=Image.open(image_name)
        label_RGB=Image.open(label_name)
        label=label_RGB.convert('L')
        
        if self.image_transform is not None:
            image=self.image_transform(image)
        if self.label_transform is not None:
            label=self.label_transform(label)
        
        return image,label

# =============================================================================
# 网络结构
# =============================================================================
#model=Deeplabv2(num_class=2)
#model=DeepLabV3(num_classes=2)
model=DeepLabv3_plus(num_classes=2)
#model=ResUnet(num_classes=2)
#model=FCN8(num_classes=2)
model.cuda()
# =============================================================================
# 训练
# =============================================================================
def train():
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=args.lr)
    train_dataset=create_dataset('/home/yzy/胃镜分割/data1/train/')
    train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=0)
    for epoch in range(1,args.epochs+1):
        model.train()
        #start time
        start=time.time()
        step=0
        epoch_loss=0
        for x,y in train_loader:
            step+=1
            
            #FiveCrop
            bs, ncrops,c ,h ,w = x.size()
            x=x.view(bs*ncrops,c,h,w)
            bs, ncrops,c ,h ,w = y.size()
            y=y.view(bs*ncrops,c,h,w)
            
            inputs=x.to(device)
            labels=y.to(device)  
            labels=labels.squeeze(1)
            #forward pass
            outputs=model(inputs)
            loss=criterion(outputs,labels.long())
            #backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            end=time.time()
            epoch_loss+=loss
        if epoch % args.display_epoch == 0:
            print(f"Epoch: [{epoch}/{args.epochs}]",
                  f"Loss:  {epoch_loss.item():.8f}",
                  f"Time: {(end-start)*args.display_epoch:.1f}sec!"
                    )
            model.eval()
            
        #Save the model checkpoint
        if epoch>=50 and epoch%10==0 :
#           torch.save(model,args.model_path+args.model_save_name)
            args.model_save_name=args.model_name+'-'+str(epoch)
            torch.save(model.state_dict(),args.model_path+args.model_save_name)
            print('Model save to ',{args.model_path+args.model_name+'-'+str(epoch)})
            
    print('Finished training!')

if args.train==True:
    action='train'
    train()
# =============================================================================
# 评估指标
# =============================================================================
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

# =============================================================================
# 测试
# =============================================================================
def test():
#    model=torch.load(args.model_path+args.model_save_name).to(device)
    epoch=[50]
    
    for i in range(1):
    
        args.model_save_name=args.model_name+'-'+str(epoch[i])
        
        model.load_state_dict(torch.load(args.model_path+args.model_save_name))
        test_dataset=create_dataset('/home/yzy/胃镜分割/data1/test/')
        test_loader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=0)
        evaluator = Evaluator(args.num_class)
        print(test_loader)
        for x,y in test_loader:
            '''                     
            #FiveCrop
            bs, ncrops,c ,h ,w = x.size()
            x=x.view(bs*ncrops,c,h,w)
            bs, ncrops,c ,h ,w = y.size()
            y=y.view(bs*ncrops,c,h,w)
            '''           
            inputs=x.to(device)
            labels=y.to(device)
            labels=labels.squeeze(1)
            outputs=model(inputs)
            outputs=Variable(outputs)
            labels=Variable(labels)
#            pred=outputs.detach().numpy()
            pred=outputs.cpu().numpy()
#            labels=labels.detach().numpy()
            labels=labels.cpu().numpy()
            pred=np.argmax(pred,1)
            evaluator.add_batch(labels, pred)
        
        # Fast test during the training
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Validation:epoch={}'.format(epoch[i]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))

if args.test==True:
    action='test'
    test()

# =============================================================================
# 验证
# =============================================================================
def validation():
    best_epoch=80
    args.model_save_name=args.model_name+'-'+str(best_epoch)
    model.load_state_dict(torch.load(args.model_path+args.model_save_name))
    valid_dataset=create_dataset('/home/yzy/胃镜分割/data1/test/')
    valid_loader=DataLoader(valid_dataset,batch_size=1,shuffle=True,num_workers=0)
    i=1
    fig=plt.gcf()
    model.eval()
    for x,y in valid_loader:
        inputs=x.to(device)
        labels=y.to(device)
        outputs=model(inputs)
        outputs=torch.argmax(outputs,1)
        outputs=Variable(outputs)
        labels=Variable(labels)
#        pred=outputs.detach().numpy()
        pred=outputs.cpu().numpy()
        pred=pred.squeeze()
        pred=pred.astype('uint8')
#        labels=labels.detach().numpy()
        labels=labels.cpu().numpy()
        labels=labels.squeeze()
        labels=labels.astype('uint8')
        plt.subplot(121)
        plt.title('prediction')
        plt.imshow(pred,cmap='gray')
        plt.subplot(122)
        plt.title('label')
        plt.imshow(labels,cmap='gray')
        fig.savefig(args.save_valid_path+'%d.png'%i)
        print('Pred save to ',args.save_valid_path+'%d.png'%i)
        i+=1
    print('Validation finished!')

if args.validation==True:
    action='validation'
    validation()
    
    


