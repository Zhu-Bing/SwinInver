from torch.cuda.amp import GradScaler, autocast
import os
import wandb
import torch
import numpy as np
from MyDataSet3D1 import seisDataset
from torch.utils.data import DataLoader
from utils.loss_class import lambda_loss
from sklearn.metrics import mean_squared_error
from FGM import FGM
import time
from torch.nn import functional as F
import random
from utils.dataAug import z_score_clip
from PGD import PGD
from matplotlib import pyplot as plt
from datetime import datetime
# from backbone.Swin_unet import SwinUNET
# from backbone.Swin_resunet import SwinResUNET
# from backbone.Resunet34 import ResUNET34
# from backbone.Swin_unetr import SwinUNETR
from backbone.Unet import UNet
from backbone.U2net import U2Net,u2net_full
# from backbone.Swinmlp_unetr import SwinMLPUNETR
# from backbone.resunet_swin import ResSwinUNET
# from backbone.SwinTransformer import SwinTransformer3D
from backbone.HRnet import HRNet
from backbone.HRNetBatchNorm import HRNetB
# from backbone.ConvHRNext import ConvHRNext
# from backbone.SwinHRnet import SwinHRNet
from backbone.SwinHRNetBatchnorm import SwinHRNetB
from backbone.dou_bankbone import HRNet as G_Net
from backbone.dou_bankbone import PixPro
# timeline=400，inline=502，crossline=501
# (400,502,501) 400个502✖501的面

# scaler = GradScaler(enabled=True)

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


if __name__ == '__main__':
    name = 'train' + datetime.now().strftime('%Y%m%d%H%M')
    wandb.init(project="Impedance", name=name)
    config = wandb.config
    config.patience = 50
    config.seed = 3
    config.seismic = 'data/denoise_seismic.npy'
    config.logCube = 'data/logCube_9log.npy'
    config.batch_size = 2
    config.base_dim = 32
    config.optim = 'AdamW'
    config.lr = 0.0005
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
    config.base_width = 32
    config.loss = 'l1'
    config.K = 10
    # config.mse_max = 12000
    # config.model = 'unet'


    all = z_score_clip(np.load(config.seismic), config.normal_clip)

    print(all.shape)
    all = torch.from_numpy(all)
    all = F.interpolate(all[None, None], (400, 528, 528), mode='trilinear', align_corners=True)

    imp = (np.load('data/imp.npy').astype(np.float32))
    val_target = imp[:, 240, :]
    plt.imshow(val_target, cmap='jet')
    target_image = wandb.Image(plt)
    wandb.log({"imp": target_image})

    a1 = np.zeros((400, 528, 528),dtype=float)
    aaa = 0

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
                            logCube_road=config.logCube,F3=False)

    train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size)
    # model = FirstNet3D(config.base_dim).to(config.device)
    # tool = G_Net(base=config.base_width)
    # model = PixPro(config, tool).to(config.device)
    # model = ResUNET(base=config.base_dim).to(config.device)# base=config.base_dim
    # model = ResUNET34(feature_size=config.base_dim).to(config.device)
    # model = u2net_full().to(config.device)
    # model = SwinResUNET().to(config.device) #feature_size=config.base_dim
    # model = UNet().to(config.device)
    # model = SwinUNETR().to(config.device)
    # model = HRNetB().to(config.device)
    # model = ConvHRNext().to(config.device)
    model = SwinHRNetB().to(config.device)
    optimizer = eval(f'torch.optim.{config.optim}')(model.parameters(), lr=config.lr)

    # checkpoint = torch.load('FGM.pth')
    # model.load_state_dict(checkpoint)

    loss_function = lambda_loss()
    pgd = PGD(model,'input_layer')
    fgm = FGM(model)
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
    mse_max = 13000
    for idx, batch in enumerate(train_loader):
        seismic, logCube= batch
        B, C, T, H, W = seismic.shape
        seis, logCube = resize_time(seismic.to(config.device)), logCube.to(config.device)# resize_time(logCube.to(config.device))
        # seis, logCube = seismic.to(config.device), logCube.to(config.device)
        model.zero_grad()
        output = model(seis,logCube)#
        train_loss = loss_function(output, logCube, config.loss)
        train_loss = train_loss.mean()
        loss = train_loss
        loss_sum.append(train_loss.item())
            # m = 1 - (1 - 0.995) * (math.cos(math.pi * idx / config.all_steps) + 1) / 2
            # ema_model = update_ema_variables(ema_model, model, m, idx)
        print('loss:', loss.item())
        loss.backward()

        # fgm.attack()
        # output_adv = model(seis,logCube)
        # train_loss_adv = loss_function(output_adv, logCube, config.loss)
        # train_loss_adv = train_loss_adv.mean()
        # loss_adv = train_loss_adv
        # print(loss_adv)
        # loss_adv.backward()
        # fgm.restore()

        pgd.backup_grad()
        for t in range(config.K):
            pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != config.K - 1:
                model.zero_grad()
            else:
                pgd.restore_grad() # 把所有的梯度还回去
            output_adv = model(seis,logCube)
            train_loss_adv = loss_function(output_adv, logCube, config.loss)
            train_loss_adv = train_loss_adv.mean()
            loss_adv = train_loss_adv
            print(loss_adv)
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        pgd.restore() # 把embedding的param.data还回去
        optimizer.step()

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
                    for i in range(1, 12):
                        for j in range(1, 12):
                            val = all[:, :, :, 48 * (i - 1):48 * i, 48 * (j - 1):48 * j].to(config.device)
                            val_output = model(val,val).cpu().numpy()[0, 0]
                            a1[:, 48 * (i - 1):48 * i, 48 * (j - 1):48 * j] = val_output

            val_target = a1[:, 240, :]
            plt.imshow(val_target, cmap='jet')
            target_image = wandb.Image(plt)
            wandb.log({"target_image": target_image})

            if idx > 900:
                a1 = torch.from_numpy(a1)
                a1 = F.interpolate(a1[None, None], (400, 502, 501), mode='trilinear', align_corners=True)
                a1 = a1.cpu().numpy()[0, 0]
                mse = mean_squared_error(imp.flatten(),a1.flatten())
                mae = np.mean(np.abs(imp - a1))
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
            a1 = np.zeros((400, 528, 528), dtype=float)
            # np.save(str(aaa)+'npy',a1)
            # if idx >30000:
            #     aaa += 1
            #     np.save(r'/root/autodl-tmp/'+str(aaa),a1)
            model.train()

        # if (idx % config.model_saved_step == 0 and idx > 20000) or (idx == config.all_steps - 1):
        #     road = os.path.join('/root/autodl-tmp',str(idx).zfill(5) + 'model.pth')
        #     torch.save(model.state_dict(), road)
        #     wandb.save(road)

    print(val_min_list)
    wandb.finish()

