import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from models.FasterRcnn.mAP import calculate_mAP, draw_PR


def train(model, device, save_path, train_dataloader, val_dataloader, test_dataloader, num_classes, learning_rate, weight_decay, epochs=10):
    os.makedirs(save_path)
    model.to(device)
    writer = SummaryWriter(save_path)
    max_iterations = epochs * len(train_dataloader)

    params = [p for p in model.parameters() if p.requires_grad]  # 仅训练可训练的参数
    optimizer = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    scheduler = custom_lr_scheduler(optimizer, lr_min=0.00001, lr_max=learning_rate, warmup_steps=int(0.1*max_iterations), total_steps=max_iterations)
    print(f'训练开始, 当前训练参数为: lr={learning_rate}, wd={weight_decay}')

    iter_num = 0
    for e in range(epochs):
        model.train()
        loop = tqdm(train_dataloader, total=len(train_dataloader))
        for image, target in loop:
            loop.set_description(f'Epoch [{e+1}/{epochs}]')
            image = [img.to(device) for img in image]
            target = [{k: v.to(device).squeeze(0) for k, v in target.items()}]

            loss_dict = model(image, target)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()
            iter_num += 1
            lr_ = optimizer.param_groups[0]['lr']
            writer.add_scalar(tag='lr', scalar_value=lr_, global_step=iter_num+1)
            writer.add_scalar(tag='Loss', scalar_value=losses.item(), global_step=iter_num+1)
            loop.set_postfix({'loss': f'{losses.item():.6f}', 'lr': f'{lr_:.6f}'})
        torch.save(model.state_dict(), f=os.path.join(save_path, 'weight.pth'))
        torch.save(model, f=os.path.join(save_path, 'model.pth'))
        print('Saved model and weights to disk.')
        mAP, _, _ = calculate_mAP(model, val_dataloader, num_classes, device, score_threshold=0)
        print(f'mAP on val_dataset: {mAP:.6f}')
    mAP, recall, precision = calculate_mAP(model, test_dataloader, num_classes, device, score_threshold=0)
    for i in range(num_classes):
        draw_PR(recall[i], precision[i], f"Class {i+1}")
    writer.close()


def custom_lr_scheduler(optimizer, lr_min, lr_max, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # 线性上升
            lr = lr_min + (lr_max - lr_min) * (current_step / warmup_steps)
        else:
            # 余弦衰减
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))
        return lr / lr_max
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
