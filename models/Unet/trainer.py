import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


# def train(model, device, dtype, optimizer_mode, trainloader, save_path, learning_rate, weight_decay=0.0001, epochs=1):
#     model = model.to(device=device)  # 将模型转移至CPU/GPU
#     writer = SummaryWriter(save_path)  # 日志保存
#     if optimizer_mode == 'SGD':
#         optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
#     elif optimizer_mode == 'Adam':
#         optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-08)
#     max_iterations = epochs * len(trainloader)
#     iter_num = 0
#     print(f'训练开始, 当前训练参数为: lr={learning_rate}, wd={weight_decay}')
#     for e in range(epochs):
#         model.train()  # 将模型设置为train模式
#         loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=True)
#         for t, sample in loop:
#             loop.set_description(f'Epoch [{e+1}/{epochs}]')
#             x = sample['image'].to(device=device, dtype=dtype)
#             y = sample['label'].to(device=device, dtype=dtype)
#             optimizer.zero_grad()
#             pred_y = model(x)

#             loss_fn = nn.CrossEntropyLoss()
#             loss = loss_fn(pred_y, y)
#             loss.backward()
#             optimizer.step()

#             if optimizer_mode == 'SGD':
#                 lr_ = learning_rate * (1.0 - iter_num / max_iterations) ** 0.9
#                 for param_group in optimizer.param_groups:
#                     param_group['lr'] = lr_

#             iter_num += 1
#             writer.add_scalar(tag='Loss', scalar_value=loss.item(), global_step=iter_num+1)
#             if optimizer_mode == 'SGD':
#                 writer.add_scalar(tag='lr', scalar_value=lr_, global_step=iter_num+1)
#             loop.set_postfix({'loss': f'{loss.item():.6f}'})

#     torch.save(model.state_dict(), f=os.path.join(save_path, 'model.pth'))
#     torch.save(model, f=os.path.join(save_path, 'full_model.pth'))
#     writer.close()


def train(model, device, save_path, train_dataloader, learning_rate, weight_decay=0.0005, epochs=10, loss_type='CrossEntropyLoss'):
    os.makedirs(save_path)
    model.to(device)
    writer = SummaryWriter(save_path)
    max_iterations = epochs * len(train_dataloader)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    scheduler = custom_lr_scheduler(optimizer, lr_min=0.000001, lr_max=learning_rate, warmup_steps=int(0.1*max_iterations), total_steps=max_iterations)
    print(f'训练开始, 当前训练参数为: lr={learning_rate}, wd={weight_decay}')

    iter_num = 0
    loss_fn = nn.CrossEntropyLoss() if loss_type == 'CrossEntropyLoss' else ModifiedMSE

    for e in range(epochs):
        model.train()
        loop = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch [{e+1}/{epochs}]')
        for image, label in loop:
            image = image.repeat(1, 3, 1, 1)
            x = image.to(device=device)
            y = label.to(device=device)
            pred_y = model(x)

            if loss_type == 'CrossEntropyLoss':
                loss = loss_fn(pred_y, y)
            else:
                pred_y = torch.sigmoid(pred_y)
                loss = loss_fn(pred_y, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            lr_ = optimizer.param_groups[0]['lr']
            writer.add_scalar(tag='lr', scalar_value=lr_, global_step=iter_num+1)
            writer.add_scalar(tag='Loss', scalar_value=loss.item(), global_step=iter_num+1)
            loop.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{lr_:.6f}'})
            iter_num += 1
        if (e+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'weight.pth'))
            torch.save(model, os.path.join(save_path, 'model.pth'))
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
    loss = torch.mean(difference / (image_gt + torch.max(image_gt) / 10 + 1e-8))
    return loss
