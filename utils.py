import torch
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def get_row_col_num(data_num):
    root = math.floor(math.sqrt(data_num))
    for row_num in range(root):
        col_num = data_num / (root - row_num)
        if math.floor(col_num) == col_num:
            return math.floor(data_num / col_num), math.floor(col_num)

def img_save(data, img_name):
    data_num = data.shape[0]
    row_num, col_num = get_row_col_num(data_num)
    fig, axes = plt.subplots(row_num, row_num, figsize=(4, 4), tight_layout=True)
    for row in range(row_num):
        for col in range(col_num):
            cur_idx = row * col_num + col
            img = data[cur_idx, :].cpu().detach()
            axes[row, col].imshow(img.numpy().reshape(28, 28), cmap='gray')
            axes[row, col].axis('off')
    plt.show()
    plt.savefig(img_name)

def generate_number_dict(data_iter):
    number_dict = {}
    for _, batch_data in enumerate(data_iter):
        batch_size = batch_data[0].shape[0]
        datas = batch_data[0]
        labels = batch_data[1]
        for data, label in zip(datas, labels):
            if label not in number_dict:
                label = label.item()
                number_dict[label] = data
        if len(number_dict) == 10:
            return number_dict

def show_number_dict(number_dict):
    for key, _ in number_dict.items():
        plt.imshow(number_dict[key].numpy().reshape(28, 28), cmap='gray')
        plt.show()
        plt.savefig('./output/' + str(key) + '.jpg')

def to_onehot(label, device):
    label_num = label.shape[0]
    label = label.unsqueeze(-1)
    one_hot_label = torch.zeros(label_num, 10).to(device)
    one_hot_label = one_hot_label.scatter_(1, label, 1)
    return one_hot_label