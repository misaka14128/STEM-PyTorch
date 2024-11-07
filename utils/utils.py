import os
import random

import numpy as np
import pandas as pd
import tensorly as tl
import hyperspy.api as hs
import matplotlib.pyplot as plt

from PIL import Image
from tensorly.decomposition import tucker
from mpl_toolkits.axes_grid1 import ImageGrid

from ase.atom import Atom
import torch.nn.functional as F


def hole(atoms, radius, size):
    x0 = random.random()*(size - 2*radius) + radius
    y0 = random.random()*(size - 2*radius) + radius
    del_list = []
    for atom in atoms:
        x = atom.get('position')[0]
        y = atom.get('position')[1]
        distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        if distance <= radius:
            del_list.append(atom.index)
    del atoms[del_list]


def random_defect(atoms, num):
    random_indices = np.random.choice(len(atoms), num, replace=False).tolist()
    positions = []
    for index in random_indices:
        positions.append(atoms[index].get('position'))
    del atoms[random_indices]
    return positions


def random_defect_te(atoms_up, atoms_down, num):
    random_indices = np.random.choice(len(atoms_up), num, replace=False).tolist()
    positions = []
    for index in random_indices:
        positions.append(atoms_up[index].get('position'))
    del atoms_up[random_indices]
    del atoms_down[random_indices]
    return positions


def generate_density_from_position(position, width, std):
    image = np.zeros((width, width))
    x, y = np.meshgrid(np.arange(width), np.arange(width))
    for i in range(position.shape[0]):
        x0 = position[i, 0]
        y0 = position[i, 1]
        distance = np.sqrt((x-x0)**2 + (y-y0)**2)
        gaussian = np.exp(-(distance**2 / (2 * std**2)))
        image += gaussian
    if np.max(image) == 0:
        return image
    image /= np.max(image)
    return image.astype(np.float32)


def get_circle_coords(img_shape, center, radius):
    x, y = center
    xx, yy = np.meshgrid(np.arange(x-radius, x+radius+1), np.arange(y-radius, y+radius+1))
    distances = np.sqrt((xx-x)**2 + (yy-y)**2)
    indices = np.where(distances <= radius)
    coords = np.column_stack((xx[indices], yy[indices]))
    coords = coords[(coords[:, 0] >= 0) & (coords[:, 1] >= 0) & (coords[:, 0] < img_shape[0]) & (coords[:, 1] < img_shape[1])]
    return coords


def generate_circle_from_position(position, size, radius):
    image = np.zeros((size, size))
    for i in range(position.shape[0]):
        center = (int(position[i, 0]), int(position[i, 1]))
        coords = get_circle_coords(image.shape, center, radius)
        for coord in coords:
            image[tuple(coord)] = 1
    return image


def xyz_to_xy(pos):
    df = pd.DataFrame(pos, columns=['col1', 'col2', 'col3'])
    new_df = df[['col1', 'col2']].drop_duplicates()
    position = new_df.values
    position[:, 0] = np.round((position[:, 0])* 10)
    position[:, 1] = np.round((position[:, 1])* 10)
    return position


def xyz_to_xy_withmono(pos):
    df = pd.DataFrame(pos, columns=['col1', 'col2', 'col3'])
    counts = df.groupby(['col1', 'col2']).size().reset_index(name='count')
    position_1 = counts[counts['count'] == 1][['col1', 'col2']].values
    position_2 = counts[counts['count'] == 2][['col1', 'col2']].values
    position_1[:, 0] = np.round((position_1[:, 0]) * 10)
    position_1[:, 1] = np.round((position_1[:, 1]) * 10)
    position_2[:, 0] = np.round((position_2[:, 0]) * 10)
    position_2[:, 1] = np.round((position_2[:, 1]) * 10)
    return position_1, position_2


def xyz_to_xy_withmono_adsorb(pos, condition_func):
    df = pd.DataFrame(pos, columns=['col1', 'col2', 'col3'])
    # 分为满足条件和不满足条件的两组
    df_condition_met = df[condition_func(df['col3'])]
    df_condition_not_met = df[~condition_func(df['col3'])]
    # 对不满足条件的坐标进行原有的分类
    counts = df_condition_not_met.groupby(['col1', 'col2']).size().reset_index(name='count')
    position_1 = counts[counts['count'] == 1][['col1', 'col2']].values
    position_2 = counts[counts['count'] == 2][['col1', 'col2']].values
    position_1[:, 0] = np.round(position_1[:, 0] * 10)
    position_1[:, 1] = np.round(position_1[:, 1] * 10)
    position_2[:, 0] = np.round(position_2[:, 0] * 10)
    position_2[:, 1] = np.round(position_2[:, 1] * 10)
    position_3 = df_condition_met[['col1', 'col2']].values
    position_3[:, 0] = np.round(position_3[:, 0] * 10)
    position_3[:, 1] = np.round(position_3[:, 1] * 10)

    # 返回满足条件的坐标和原有分类的坐标
    return position_1, position_2, position_3


def generate_background(label):
    label_sum = np.sum(label, axis=0)
    label_sum /= np.max(label_sum)
    return np.ones_like(label_sum) - label_sum


def random_process_MoTe2(atoms):
    atoms_mo = atoms[[atom.index for atom in atoms if atom.symbol=='Mo']].copy()
    atoms_te = atoms[[atom.index for atom in atoms if atom.symbol=='Te']].copy()
    num = int(0.05 * len(atoms_te))
    random_indices_te = np.random.choice(len(atoms_te), num, replace=False).tolist()
    random_indices_mo = np.random.choice(len(atoms_mo), num, replace=False).tolist()
    positions_mo = atoms_mo[random_indices_mo].arrays['positions']
    for i in range(num):
        atom = atoms_te[random_indices_te[i]]
        old_position = atom.get('position')
        del atoms_te[random_indices_te[i]]
        new_position = positions_mo[i]
        if old_position[2] > new_position[2]:
            new_position[2] = old_position[2]+0.3
        else:
            new_position[2] = old_position[2]-0.3
        atoms_te.append(Atom(symbol='Te', position=new_position))
    atoms_mo.extend(atoms_te)
    return atoms_mo


def dis(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def model_test(model, num_classes, name_list, test_input, test_label, device='cpu', learning_rate=None, weight_decay=None, save_file=None, tag=None):
    model.eval()
    score = model(test_input)
    score = F.softmax(score, dim=1)
    if device == 'cuda':
        score = score.cpu().detach().numpy()
        test_input = test_input.cpu().numpy()
        test_label = test_label.cpu().numpy()
    elif device == 'cpu':
        score = score.detach().numpy()
        test_input = test_input.numpy()
        test_label = test_label.numpy()
    else:
        print('无效的输入')
        return

    print_list = []
    print_list.append({'data': test_input[0, 0], 'name': 'input_image'})
    for i in range(num_classes):
        print_list.append({'data': test_label[0, i], 'name': 'label_'+name_list[i]})
    print_list.append({'data': test_input[0, 0], 'name': 'input_image'})
    for i in range(num_classes):
        print_list.append({'data': score[0, i], 'name': 'pred_'+name_list[i]})

    fig = plt.figure(figsize=(20, 10))
    ncols = num_classes+1
    grid = ImageGrid(fig, 111, nrows_ncols=(2, ncols), axes_pad=0.5, cbar_mode="each", cbar_size="5%", cbar_pad="2%")

    for i in range(ncols*2):
        data, name = print_list[i]['data'], print_list[i]['name']
        im = grid[i].imshow(data, cmap='gray')
        grid[i].invert_yaxis()
        grid[i].set_title(name)
        grid.cbar_axes[i].colorbar(im)

    if tag is not None and learning_rate is not None and weight_decay is not None and save_file is not None:
        plt.suptitle(f'parameters: lr={learning_rate}, wd={weight_decay}, tag:{tag}')
        plt.savefig(os.path.join(save_file, f'test_{tag}.jpg'))
    plt.show()


def denoise(EMdata, subrow=64, subcol=64, step_size=16):
    sub3D = []
    overlap = np.zeros_like(EMdata)
    row_num = int((EMdata.shape[0]-subrow)/step_size+1)
    col_num = int((EMdata.shape[1]-subcol)/step_size+1)
    for i in range(row_num):
        for j in range(col_num):
            overlap[i*step_size:i*step_size+subrow, j*step_size:j*step_size+subcol] += 1
            sub3D.append(EMdata[i*step_size:i*step_size+subrow, j*step_size:j*step_size+subcol].copy())
    sub3D = np.asarray(sub3D)
    tucker_rank = [40, int(subrow/2), int(subcol/2)]
    core, tucker_factors = tucker(np.float64(sub3D), rank=tucker_rank, init='random', tol=10e-9)
    denoised3D = tl.tucker_to_tensor((core, tucker_factors))
    denoisedEM = np.zeros(EMdata.shape, denoised3D.dtype)
    for i in range(row_num):
        for j in range(col_num):
            denoisedEM[i*step_size:i*step_size+subrow, j*step_size:j*step_size+subcol] += denoised3D[i*col_num+j, :, :]
    denoisedEM /= overlap
    return data_init(np.float32(denoisedEM))


def load_dm3(filename):
    return np.float32(np.array(hs.load(filename)))


def save_as_image(array, path):
    arr_min = np.min(array)
    arr_max = np.max(array)
    array = ((array-arr_min) / (arr_max - arr_min)) * 255
    array = array.astype(np.uint8)
    image = Image.fromarray(array)
    image.save(path)


def trans_pos_to_label(pos_list, width):
    label_list = []
    for item in pos_list:
        added_pos = item + width/2
        subtracted_pos = item - width/2
        final_pos = np.hstack((subtracted_pos, added_pos))
        label_list.append(final_pos)
    return label_list


def data_init(data):
    return (data - np.min(data))/(np.max(data) - np.min(data))


if __name__ == '__main__':
    pass
