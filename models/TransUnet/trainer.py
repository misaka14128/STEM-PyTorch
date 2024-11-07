import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


def train(model, device, save_path, train_dataloader, learning_rate, weight_decay=0.0005, epochs=10, loss_type='CrossEntropyLoss'):
    os.makedirs(save_path)
    model.to(device)
    writer = SummaryWriter(save_path)
    max_iterations = epochs * len(train_dataloader)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    scheduler = custom_lr_scheduler(optimizer, lr_min=0.00001, lr_max=learning_rate, warmup_steps=int(0.1*max_iterations), total_steps=max_iterations)
    print(f'训练开始, 当前训练参数为: lr={learning_rate}, wd={weight_decay}')

    iter_num = 0
    for e in range(epochs):
        model.train()
        loop = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch [{e+1}/{epochs}]')
        for image, label in loop:
            x = image.to(device=device)
            y = label.to(device=device)
            pred_y = model(x)

            if loss_type == 'CrossEntropyLoss':
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(pred_y, y)
            elif loss_type == 'ModifiedMSE':
                pred_y = torch.sigmoid(pred_y)
                loss = ModifiedMSE(pred_y, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr_ = optimizer.param_groups[0]['lr']
            iter_num += 1
            writer.add_scalar(tag='lr', scalar_value=lr_, global_step=iter_num+1)
            writer.add_scalar(tag='Loss', scalar_value=loss.item(), global_step=iter_num+1)
            loop.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{lr_:.6f}'})
        torch.save(model.state_dict(), f=os.path.join(save_path, 'weight.pth'))
        torch.save(model, f=os.path.join(save_path, 'model.pth'))
        print('Saved model and weights to disk.')
    writer.close()


def custom_lr_scheduler(optimizer, lr_min, lr_max, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            lr = lr_min + (lr_max - lr_min) * (current_step / warmup_steps)
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))
        return lr / lr_max
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def ModifiedMSE(image, image_gt):  # (N, C, H, W)
    difference = (image - image_gt) ** 2
    loss = torch.mean(difference / (image_gt + torch.max(image_gt) / 10))
    return loss
