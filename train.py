import torch
from torch.autograd import Variable
import os
from datetime import datetime
from torch.cuda import device
from model.LENS_Net import LENS_Net
from utils.data_loader import get_loader
from utils.larning import clip_gradient, adjust_lr
import yaml
from spikingjelly.clock_driven import functional

filename = "config/LENS_Net.yaml"  #load config file
try:
    with open(filename, 'r', encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)
        print(yaml_data)
except FileNotFoundError:
    print(f"File '{filename}' not found.")
except yaml.YAMLError as e:
    print(f"Error while loading YAML: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

epochs = yaml_data["train"]["epoch"]
lr = 1e-4
batchsize = yaml_data["train"]["batchsize"]
trainsize = yaml_data["train"]["trainsize"]
clip = yaml_data["train"]["clip"]
decay_rate = yaml_data["train"]["decay_rate"]
decay_epoch = yaml_data["train"]["decay_epoch"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = LENS_Net().to(device)



print(torch.__version__)

params = model.parameters()
optimizer = torch.optim.Adam(params, lr) #



image_root = './datasets/train_dataset/ORSSD/image/'
gt_root = './datasets/train_dataset/ORSSD/GT/'
train_loader = get_loader(image_root, gt_root, batchsize=batchsize, trainsize=trainsize)
total_step = len(train_loader)


BCE = torch.nn.BCEWithLogitsLoss()
from utils.utils import IOU
save_path = 'logs/LENS_Net/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def train(train_loader, model, optimizer, epoch, Epochs):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()  #提取清空为0

        functional.reset_net(model)


        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.to(device)
        gts = gts.to(device)

        p= model(images) #return features from encoder and decoder
        loss = 0.0
        w_ce, w_dice = 0.3, 0.7
        for j in range(4):
            loss_ce = BCE(p[j], gts)

            p_sig = torch.sigmoid(p[j])
            loss_dice = IOU(p_sig, gts)  # design a hybrid loss
            loss += (w_ce * loss_ce + w_dice * loss_dice)
        loss.backward(retain_graph=True)


        clip_gradient(optimizer, clip)

        optimizer.step()

        # torch.save(model.state_dict(), save_path + f'fepoch{epoch}_SpikeNetv8.pth')

        if i % 20 == 0 or i == total_step:
            print(
                '{} epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}'.
                    format(datetime.now(), epoch, Epochs, i, total_step,
                           lr * decay_rate ** (epoch // decay_epoch), loss.data))


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # if epoch % 5 == 0:
    #     torch.save(model.state_dict(), save_path + f'f{epoch}GeleNet.pth')
    torch.save(model.state_dict(), save_path + f'epoch{epoch}_LENS_Net.pth')


print("begin to train！")
for i in range(1, epochs+1):
    adjust_lr(optimizer, lr, i, decay_rate, decay_epoch) #adjust lr
    train(train_loader, model, optimizer, i, epochs)

