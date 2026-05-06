from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torchvision import datasets, transforms
import numpy as np
from datetime import datetime

import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import torch

def visualize(feat, labels, epoch, writer, name, dpi=100):
    # 提亮版 tab10
    bright_tab10 = [
    "#5dade2", # 浅蓝  (原 #1f77b4)
    "#ffbb78", # 浅橙  (原 #ff7f0e)
    "#58d68d", # 浅绿  (原 #2ca02c)
    "#ff7b7b", # 浅红  (原 #d62728)
    "#b39ddb", # 浅紫  (原 #9467bd)
    "#a67c52", # 浅棕  (原 #8c564b)
    "#f7b6d2", # 浅粉  (原 #e377c2)
    "#b2babb", # 浅灰  (原 #7f7f7f)
    "#d4ef4a", # 黄绿  (原 #bcbd22)
    "#76d7c4", # 浅青  (原 #17becf)
    ]


    colors = bright_tab10
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16})

    # 1. 画布 10×9 inch
    fig = Figure(figsize=(10, 9), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    # 2. 数据散点 + 数字
    feat = feat.data.cpu().numpy()
    labels = labels.data.cpu().numpy()

    for i in range(10):
        idx = (labels == i)
        ax.scatter(feat[idx, 0], feat[idx, 1],
                   color=colors[i], s=15, alpha=0.85,
                   edgecolor='k', linewidth=0.5)
        if np.any(idx):
            ax.text(feat[idx, 0].mean(), feat[idx, 1].mean(), str(i),
                    fontsize=48, fontweight='bold',
                    color=colors[i], ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7,
                              edgecolor='none', pad=1))

    # 3. 轴外观
    ax.tick_params(axis='both', which='both', labelsize=32)
    sci_fmt = ScalarFormatter(useMathText=True)
    sci_fmt.set_powerlimits((-2, 2))
    sci_fmt.set_scientific(True)
    sci_fmt.set_useOffset(False)
    ax.xaxis.set_major_formatter(sci_fmt)
    ax.yaxis.set_major_formatter(sci_fmt)
    # 科学计数法的乘数文本字号
    ax.xaxis.get_offset_text().set_fontsize(32)
    ax.yaxis.get_offset_text().set_fontsize(32)

    ax.spines[['right', 'top']].set_visible(False)

    # —— 3.1   调整坐标范围保持 10:9 比例 ——
    desired = 10 / 9
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    w, h = (x_max - x_min), (y_max - y_min)
    if w / h > desired:        # x 过宽，扩 y
        new_half = w / desired / 2
        y_c = (y_min + y_max) / 2
        ax.set_ylim(y_c - new_half, y_c + new_half)
    else:                      # y 过高，扩 x
        new_half = h * desired / 2
        x_c = (x_min + x_max) / 2
        ax.set_xlim(x_c - new_half, x_c + new_half)
    ax.set_aspect('equal', adjustable='box')

    # —— 3.2   极小留白（替代 ax.margins） ——
    ax.margins(0.02)    # 对称 2%，避免点贴边

    # —— 3.3   压缩 subplot 边界，上/右 几乎 0 留白 ——
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.97, top=0.97)

    # 4. 保存 PDF（10×9 inch，比例不变）
    pdf_dir = "E:/FR-Loss-on-Mnist-master/log/figure_f/"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{name}_epoch{epoch:03d}.jpg")
    fig.savefig(pdf_path, format="jpg", pad_inches=0.02)

    # 5. TensorBoard
    canvas.draw()
    w_px, h_px = fig.get_size_inches() * fig.get_dpi()
    img = (np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
             .reshape(int(h_px), int(w_px), 4)[..., 1:])
    writer.add_image(name, transforms.ToTensor()(img), epoch)

    plt.close(fig)

def visualize_center(centers,feat, labels, epoch,writer,name):
    colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
              '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    fig = Figure(figsize=(6, 6), dpi=100)
    fig.clf()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    feat = feat.data.cpu().numpy()
    labels = labels.data.cpu().numpy()

    for i in range(10):
        ax.scatter(feat[labels == i, 0], feat[labels == i, 1], c=colors[i], s=1)
        ax.text(centers[i, 0], centers[i, 1], 'c' + str(i), color='black', fontsize=12)
        #ax.text(feat[labels == i, 0].mean(), feat[labels == i, 1].mean(), str(i), color='black', fontsize=12)
    ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    ax.text(0, 0, "epoch=%d" % epoch)
    canvas.draw()

    # if (os.path.exists(imgDir)):
    #     pass
    # else:
    #     os.makedirs(imgDir)
    # fig.savefig(imgDir + '/epoch=%d.jpg' % epoch)
    width, height = fig.get_size_inches() * fig.get_dpi()
    # img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    img = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(int(height), int(width), 4)[..., 1:] 

    tt = transforms.ToTensor()
    timg = tt(img)
    timg.unsqueeze(0)
    writer.add_image(name, timg, epoch)

def visualize_cos(feat, labels, epoch, writer, head, name,
                  dpi=100,
                  pdf_dir="E:/FR-Loss-on-Mnist-master/log/figure_w/"):

    # 0. 更亮的 tab10 配色
    bright_tab10 = [
        "#5dade2", # 浅蓝  (原 #1f77b4)
        "#ffbb78", # 浅橙  (原 #ff7f0e)
        "#58d68d", # 浅绿  (原 #2ca02c)
        "#ff7b7b", # 浅红  (原 #d62728)
        "#b39ddb", # 浅紫  (原 #9467bd)
        "#a67c52", # 浅棕  (原 #8c564b)
        "#f7b6d2", # 浅粉  (原 #e377c2)
        "#b2babb", # 浅灰  (原 #7f7f7f)
        "#d4ef4a", # 黄绿  (原 #bcbd22)
        "#76d7c4", # 浅青  (原 #17becf)
    ]
    colors = bright_tab10
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16})

    # 1. 画布 10×9 inch
    fig = Figure(figsize=(10, 9), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    # 2. 权重向量
    with torch.no_grad():
        weight = head.state_dict()['weight'].t().cpu().numpy()   # (10, 2)

    # 3. 箭头 + 数字
    for i in range(10):
        ax.quiver(0, 0, weight[i, 0], weight[i, 1],
                  angles='xy', scale_units='xy', scale=1,
                  color=colors[i], width=0.006)
        ax.text(weight[i, 0]*1.08, weight[i, 1]*1.08, str(i),
                fontsize=40, fontweight='bold', color=colors[i],
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7,
                          edgecolor='none', pad=1))

    # 4. 坐标范围适配 10:9
    lim = max(np.max(np.abs(weight)*1.08), 1e-3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    desired = 10/9
    w = (ax.get_xlim()[1] - ax.get_xlim()[0])
    h = (ax.get_ylim()[1] - ax.get_ylim()[0])
    if w/h > desired:                 # 扩 y
        half = w/desired/2
        yc   = np.mean(ax.get_ylim())
        ax.set_ylim(yc-half, yc+half)
    else:                             # 扩 x
        half = h*desired/2
        xc   = np.mean(ax.get_xlim())
        ax.set_xlim(xc-half, xc+half)

    ax.set_aspect('equal', adjustable='box')
    ax.spines[['right', 'top']].set_visible(False)

    # 5. 刻度 & 科学计数
    ax.tick_params(axis='both', which='both', labelsize=32)
    sci = ScalarFormatter(useMathText=True); sci.set_powerlimits((-2, 2))
    sci.set_scientific(True); sci.set_useOffset(False)
    ax.xaxis.set_major_formatter(sci); ax.yaxis.set_major_formatter(sci)

    # ★ 6. 极小留白 + 压缩上/右边距
    ax.margins(0.02)                              # 对称 2% 留白
    fig.subplots_adjust(left=0.08, bottom=0.08,   # 给刻度标签空间
                        right=0.995, top=0.995)   # 右/上 几乎贴边

    # 7. 保存 PDF（10×9 inch）
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{name}_cos_epoch{epoch:03d}.jpg")
    fig.savefig(pdf_path, format="jpg", pad_inches=0.02)

    # 8. TensorBoard
    canvas.draw()
    w_px, h_px = fig.get_size_inches()*fig.get_dpi()
    img = (np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
             .reshape(int(h_px), int(w_px), 4)[..., 1:])
    writer.add_image(f"{name}_cos", transforms.ToTensor()(img), epoch)

    plt.close(fig)


def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')
