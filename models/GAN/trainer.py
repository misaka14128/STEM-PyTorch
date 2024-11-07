import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


def train(generator, discriminator, device, save_path, train_dataloader, learning_rate_g=0.0002, learning_rate_d=0.0002, epochs=10):
    os.makedirs(save_path, exist_ok=True)
    generator.to(device)
    discriminator.to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate_g, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=(0.5, 0.999))
    g_scheduler = LinearDecayLR(g_optimizer, 50, epochs, learning_rate_g, 1e-6)
    d_scheduler = LinearDecayLR(d_optimizer, 50, epochs, learning_rate_d, 1e-6)
    criterion = nn.BCELoss()
    L1_loss = nn.L1Loss()
    writer = SummaryWriter(save_path)
    step_tracker = StepTracker()
    # register_gradient_hook(writer, generator, 'conv1.0.weight', 'Generator_top', step_tracker)
    # register_gradient_hook(writer, generator, 'upconv3.weight', 'Generator_bottom', step_tracker)
    # register_gradient_hook(writer, discriminator, 'model.0.weight', 'Discriminator_top', step_tracker)
    # register_gradient_hook(writer, discriminator, 'model.9.weight', 'Discriminator_bottom', step_tracker)
    print(f'GAN训练开始, 当前训练参数为: lr_g={learning_rate_g}, lr_d={learning_rate_d}')
    iter_num = 0
    for e in range(epochs):
        generator.train()
        discriminator.train()
        loop = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch [{e+1}/{epochs}]')
        for image, label in loop:
            noisy_images = image.to(device)  # (N, 1, H, W)
            clean_images = label.to(device)  # (N, 1, H, W)
            # ============ 训练判别器 ============
            # 使用真实图像
            d_optimizer.zero_grad()
            real_images = clean_images
            outputs = discriminator(real_images, noisy_images)  # (N, 2, H, W) -> (N, 1, H/8, W/8)
            real_labels = torch.ones_like(outputs, device=device)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            # 使用假图像
            fake_images = generator(noisy_images).detach()
            outputs = discriminator(fake_images, noisy_images)  # 从生成器生成假图像 (N, 2, H, W) -> (N, 1, H/8, W/8)
            fake_labels = torch.zeros_like(outputs, device=device)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            # 判别器的损失
            d_loss = (d_loss_real + d_loss_fake)/2
            # 反向传播和优化
            d_loss.backward()
            d_optimizer.step()

            # ============ 训练生成器 ============
            # 生成假图像
            g_optimizer.zero_grad()
            fake_images = generator(noisy_images)
            outputs = discriminator(fake_images, noisy_images)
            # real_labels = torch.ones_like(outputs, device=device)
            # g_loss_BCE = criterion(outputs, real_labels)

            # 生成器的损失
            g_loss = -torch.mean(torch.log(outputs+1e-8)) + 100*L1_loss(fake_images, clean_images)
            # 反向传播和优化
            g_loss.backward()
            g_optimizer.step()

            iter_num += 1
            step_tracker.increment()
            writer.add_scalar(tag='d_loss', scalar_value=d_loss.item(), global_step=iter_num+1)
            writer.add_scalar(tag='g_loss', scalar_value=g_loss.item(), global_step=iter_num+1)
            writer.add_scalar(tag='D(x)', scalar_value=real_score.mean().item(), global_step=iter_num+1)
            writer.add_scalar(tag='D(G(z))', scalar_value=fake_score.mean().item(), global_step=iter_num+1)
            loop.set_postfix({'d_loss': f'{d_loss.item():.4f}', 'g_loss': f'{g_loss.item():.4f}', 'D(x)': f'{real_score.mean().item():.4f}', 'D(G(z))': f'{fake_score.mean().item():.4f}'})
        g_scheduler.step()
        d_scheduler.step()
        torch.save(generator.state_dict(), os.path.join(save_path, f'{e}_weight.pth'))
        torch.save(discriminator.state_dict(), os.path.join(save_path, f'{e}_discriminator.pth'))
        print('Saved generator and discriminator to disk.')
    writer.close()


# def train_WGANGP(generator, discriminator, device, save_path, train_dataloader, learning_rate_g=0.00005, learning_rate_d=0.00005, epochs=10, critic_iters=5, lambda_gp=10):
#     os.makedirs(save_path, exist_ok=True)
#     generator.to(device)
#     discriminator.to(device)
#     g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=learning_rate_g)
#     d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=learning_rate_d)
#     writer = SummaryWriter(save_path)
#     step_tracker = StepTracker()
#     register_gradient_hook(writer, generator, 'conv1.0.weight', 'Generator_top', step_tracker)
#     register_gradient_hook(writer, generator, 'upconv3.weight', 'Generator_bottom', step_tracker)
#     register_gradient_hook(writer, discriminator, 'model.0.weight', 'Discriminator_top', step_tracker)
#     register_gradient_hook(writer, discriminator, 'model.6.weight', 'Discriminator_bottom', step_tracker)
#     print(f'GAN训练开始, 当前训练参数为: lr_g={learning_rate_g}, lr_d={learning_rate_d}')
#     iter_num = 0
#     for e in range(epochs):
#         generator.train()
#         discriminator.train()
#         loop = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch [{e+1}/{epochs}]')
#         for image, label in loop:
#             noisy_images = image.to(device)  # (N, 1, H, W)
#             clean_images = label.to(device)  # (N, 1, H, W)
#             # ============ 训练判别器 ============
#             for _ in range(critic_iters):
#                 discriminator.zero_grad()
#                 real_outputs = discriminator(clean_images)
#                 fake_images = generator(noisy_images).detach()
#                 fake_outputs = discriminator(fake_images)
#                 d_loss = -torch.mean(real_outputs) + torch.mean(fake_outputs)
#                 gradient_penalty = compute_gradient_penalty(discriminator, clean_images, fake_images, device)
#                 d_loss += lambda_gp * gradient_penalty
#                 d_loss.backward()
#                 d_optimizer.step()
#             # ============ 训练生成器 ============
#             generator.zero_grad()
#             fake_images = generator(noisy_images)
#             gen_outputs = discriminator(fake_images)
#             g_loss = -torch.mean(gen_outputs)
#             g_loss.backward()
#             g_optimizer.step()

#             iter_num += 1
#             step_tracker.increment()
#             writer.add_scalar(tag='d_loss', scalar_value=d_loss.item(), global_step=iter_num+1)
#             writer.add_scalar(tag='g_loss', scalar_value=g_loss.item(), global_step=iter_num+1)
#             loop.set_postfix({'d_loss': f'{d_loss.item():.4f}', 'g_loss': f'{g_loss.item():.4f}'})
#         torch.save(generator.state_dict(), os.path.join(save_path, f'{e}_weight.pth'))
#         torch.save(discriminator.state_dict(), os.path.join(save_path, f'{e}_discriminator.pth'))
#         print('Saved generator and discriminator to disk.')
#     writer.close()


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """ 计算梯度惩罚 """
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.shape, requires_grad=False, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def register_gradient_hook(writer, model, layer_name, tag_prefix, step_tracker):
    layer = dict([*model.named_parameters()])[layer_name]

    def hook(grad):
        grad_mean = grad.abs().mean()
        currrent_step = step_tracker.get_step()
        writer.add_scalar(tag=f'{tag_prefix}_grad', scalar_value=grad_mean.item(), global_step=currrent_step)
    layer.register_hook(hook)


class StepTracker:
    def __init__(self):
        self.step = 0

    def increment(self):
        self.step += 1

    def get_step(self):
        return self.step


class LinearDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, decay_start_epoch, total_epochs, start_lr, end_lr, last_epoch=-1):
        self.decay_start_epoch = decay_start_epoch
        self.total_epochs = total_epochs
        self.start_lr = start_lr
        self.end_lr = end_lr
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch
        if current_epoch < self.decay_start_epoch:
            # 如果当前 epoch 小于开始衰减的 epoch，保持初始学习率
            return [self.start_lr for _ in self.optimizer.param_groups]
        else:
            # 计算衰减后的学习率
            total_decay_epochs = self.total_epochs - self.decay_start_epoch
            lr_decay = (self.start_lr - self.end_lr) / total_decay_epochs
            decayed_epochs = current_epoch - self.decay_start_epoch
            new_lr = self.start_lr - decayed_epochs * lr_decay
            return [new_lr for _ in self.optimizer.param_groups]
