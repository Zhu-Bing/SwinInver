from torch.cuda.amp import GradScaler, autocast
import os
import wandb
import torch
import numpy as np
from MyDataSet3D1 import seisDataset
from torch.utils.data import DataLoader
from utils.loss_class import lambda_loss
from sklearn.metrics import mean_squared_error
import time
from torch.nn import functional as F
import random
from utils.dataAug import z_score_clip
from matplotlib import pyplot as plt
from datetime import datetime
from backbone.Swin_unet import SwinUNET
from backbone.Swin_resunet import SwinResUNET
# from backbone.Resunet34 import ResUNET
from backbone.Swin_unetr import SwinUNETR
from backbone.Unet import UNet
from backbone.U2net import U2Net,u2net_full
from backbone.Swinmlp_unetr import SwinMLPUNETR
from backbone.resunet_swin import ResSwinUNET
from backbone.SwinTransformer import SwinTransformer3D
from backbone.HRnet import HRNet
from backbone.SwinHRnet import SwinHRNet
from backbone.HRNetBatchNorm import HRNetB
from backbone.SwinHRNetBatchnorm import SwinHRNetB
from backbone.res_unet import ResUnet
# timeline=400，inline=502，crossline=501
# (400,502,501) 400个502✖501的面

scaler = GradScaler(enabled=True)

def resize_time(seis):
    if random.random() < 0.2:
        return seis
    if random.randint(0, 1):
        time_scale = random.uniform(1, 2)
    else:
        time_scale = random.uniform(0.5, 1)
    resize_time = int(round(T * time_scale))
    resize_time = resize_time + 16 - (resize_time % 16)
    seis = F.interpolate(seis, (resize_time, H, W), mode='trilinear', align_corners=True)
    return seis

# def resize_time(seis):
#     # if random.random() < 0.2: return seis
#     # if random.randint(0, 1):
#     #     time_scale = random.uniform(1, 2)
#     # else:
#     # time_scale = random.uniform(0.5, 1)
#     resize_time = 448
#     # int(round(T * time_scale) + (56 - (round(T * time_scale) % 56)))
#     seis = F.interpolate(seis, (resize_time, H, W), mode='trilinear', align_corners=True)
#     return seis

def logCubenorm(logCube):
    logCube = np.where(logCube != -999, np.clip(logCube, a_min=3500, a_max=6000),logCube)
    MAX = np.max(logCube)
    MIN = np.min(np.where(logCube == -999, 9999999, logCube))
    logVec = np.where(logCube != -999, (logCube - MIN) / (MAX - MIN), logCube)
    print(MAX,MIN,np.min(logCube))
    return logVec,MAX,MIN

if __name__ == '__main__':
    name = 'train' + datetime.now().strftime('%Y%m%d%H%M')
    wandb.init(project="Impedance", name=name)
    config = wandb.config
    config.patience = 50
    config.seed = 3
    config.seismic = 'data/F3/ORGSeis.npy'
    config.logCube = 'data/F3/AI.npy'
    config.batch_size = 4
    config.base_dim = 32
    config.optim = 'AdamW'
    config.lr = 0.0001
    config.all_steps = 50000
    config.restore = None
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.m_cover = 5
    config.model_saved_step = 1000
    config.normal_clip = 3.2
    config.train_cube_size = 48
    config.crop_lim = (0.5, 2)
    config.loss_saved_step = 10
    config.val_step = 1000
    config.slice = 24
    config.base_width = 24
    config.loss = 'l1'
    # config.model = 'unet'


    all = z_score_clip(np.load(config.seismic), config.normal_clip)
    print(all.shape)
    all = torch.from_numpy(all)
    all = F.interpolate(all[None, None], (400, 672, 960), mode='trilinear', align_corners=True)
    a1 = np.zeros((400, 672, 960),dtype=float)
    aaa = 0

    imp = np.load(config.logCube)

    imp1 = np.where(imp != -999, np.clip(imp, a_min=3500, a_max=6000),imp)
    MAX = np.max(imp1)
    MIN = np.min(np.where(imp1 == -999, 9999999, imp1))
    logVec = np.where(imp1 != -999, (imp1 - MIN) / (MAX - MIN), imp1)
    v1 = logVec[:, 143, 87]
    v2 = logVec[:, 144, 86]
    v3 = logVec[:, 144, 87]
    v4 = logVec[:, 144, 88]
    v5 = logVec[:, 145, 87]


    # val = all[:, 0:48, 0:48]
    #
    # val_target = val[:, config.slice, :]
    # plt.imshow(val_target, cmap='seismic')
    # target_image = wandb.Image(plt)
    # wandb.log({"target_image": target_image})
    # plt.show()

    # imp = (np.load('data/imp.npy').astype(np.float32))
    # # print(imp.shape)
    # imp = imp[:, 0:48, 0:48]
    # imp_t = imp[:, config.slice, :]
    # plt.imshow(imp_t, cmap='seismic')
    # inver = wandb.Image(plt)
    # wandb.log({"inver": inver})
    # plt.show()

    random.seed(config.seed)
    np.random.seed(config.seed)
    train_set = seisDataset(config, iters=config.batch_size * config.all_steps, seismic_road=config.seismic,
                            logCube_road=config.logCube,F3=True)

    train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size)
    # model = FirstNet3D(config.base_dim).to(config.device)
    # tool = G_Net(base=config.base_width)
    # model = PixPro(config, tool).to(config.device)
    model = HRNetB(base=config.base_dim).to(config.device)# base=config.base_dim
    # model = HRNetB(base=config.base_dim).to(config.device)
    # model = ResUnet().to(config.device)
    # model = u2net_full().to(config.device)
    # model = UNet().to(config.device)
    optimizer = eval(f'torch.optim.{config.optim}')(model.parameters(), lr=config.lr)
    loss_function = lambda_loss()
    # model = torch.nn.DataParallel(model)
    # ema_model = create_ema_model(model)
    wandb.watch(model, log="all")

    if config.restore is not None:
        restore = torch.load(config.restore)
        model.load_state_dict(restore['net'], strict=False)
        # optimizer.load_state_dict(restore['optimizer'])
        # amp.load_state_dict(restore['amp'])
        print(f'load {config.restore} success!')

    print('==============Pretrain step=================')
    s_time = time.time()
    model.train()
    loss_sum = []
    val_loss_sum = []
    val_min = 0.005
    val_min_list = []
    mse_max = 1e9
    for idx, batch in enumerate(train_loader):
        seismic, logCube= batch
        B, C, T, H, W = seismic.shape
        seis, logCube = resize_time(seismic.to(config.device)), logCube.to(config.device)# resize_time(logCube.to(config.device))
        optimizer.zero_grad()
        with autocast(enabled=True):
            output = model(seis,logCube)#
            train_loss = loss_function(output, logCube, config.loss)
            train_loss = train_loss.mean()
            loss = train_loss
            loss_sum.append(train_loss.item())
            # m = 1 - (1 - 0.995) * (math.cos(math.pi * idx / config.all_steps) + 1) / 2
            # ema_model = update_ema_variables(ema_model, model, m, idx)
        print('loss:', loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if idx % config.loss_saved_step == 0 and idx != 0:
            e_time = time.time()
            int_time = e_time - s_time
            print("Iteration:%d, Loss:%.6f,time_taken:%.2f" % (idx, np.mean(loss_sum), int_time))
            s_time = time.time()
            wandb.log({
                "train_loss": np.mean(loss_sum)
            })
            loss_sum = []

        if (idx % config.val_step == 0 and idx != 0 ) or (idx == config.all_steps - 1):
            model.eval()
            with torch.no_grad():
                with autocast(enabled=True):
                    for i in range(1, 15):
                        for j in range(1, 21):
                            val = all[:, :, :, 48 * (i - 1):48 * i, 48 * (j - 1):48 * j].to(config.device)
                            val_output = model(val,val).cpu().numpy()[0, 0]
                            a1[:, 48 * (i - 1):48 * i, 48 * (j - 1):48 * j] = val_output

            val_target = a1[:, 240, :]
            plt.imshow(val_target, cmap='jet')
            target_image = wandb.Image(plt)
            wandb.log({"target_image": target_image})

            if idx > 2000:
                a1 = torch.from_numpy(a1)
                a1 = F.interpolate(a1[None, None], (400, 651, 951), mode='trilinear', align_corners=True)
                a1 = a1.cpu().numpy()[0, 0]
                p1 = a1[:, 143, 87]
                p2 = a1[:, 144, 86]
                p3 = a1[:, 144, 87]
                p4 = a1[:, 144, 88]
                P5 = a1[:, 145, 87]
                mse = (np.sum((p1 - v1)**2) / 400 + np.sum((p2 - v2)**2) / 400 + np.sum((p3 - v3)**2) / 400 + np.sum((p4 - v4)**2) / 400)/4
                mae = (np.sum(np.abs(p1 - v1)) / 400 + np.sum(np.abs(p2 - v2)) / 400 + np.sum(np.abs(p3 - v3)) / 400+ np.sum(np.abs(p4 - v4)) / 400)/4

                wandb.log({
                    "mse": mse,
                    "mae": mae
                })
                print('mse:',mse)
                print('mae:',mae)
                if mse_max > mse:
                    mse_max = mse
                    road = os.path.join('/root/autodl-tmp', str(mse) + 'model.pth')
                    torch.save(model.state_dict(), road)
                    wandb.save(road)
            a1 = np.zeros((400, 672, 960), dtype=float)
            # np.save(str(aaa)+'npy',a1)
            # if idx >30000:
            #     aaa += 1
            #     np.save(r'/root/autodl-tmp/'+str(aaa),a1)
            model.train()
    print(val_min_list)
    wandb.finish()

