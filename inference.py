from model.LENS_Net import LENS_Net
import torch
import torch.nn.functional as F
import imageio
import torchvision.transforms as transforms
import  random

def set_seed(seed=42):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(3407)  # 一行调用

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LENS_Net().to(device)
state_dict = torch.load('logs/LENS_Net/xxx.pth', map_location="cpu")
model.load_state_dict(state_dict, strict=False)  # 当加载的权重与结果不匹配时就会出现错误。
params = model.parameters()
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from torch.utils.tensorboard import SummaryWriter
#
# writer = SummaryWriter(log_dir='./logs/neuron_data')

dataset_path = './datasets/test_dataset/'

testsize = 352

time_sum = 0
model.eval()

test_datasets = ['EORSSD']
import os
from PIL import Image
import numpy as np
def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def img_transform(image):
    transform_pipeline = transforms.Compose([
        transforms.Resize((352, 352)),  # 调整大小
        transforms.ToTensor(),  # 转为 Tensor（并归一化到 [0, 1]）
        transforms.Normalize(  # 标准化（均值、方差）
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform_pipeline(image)  # 应用转换


def transform_image(image):
    image = rgb_loader(image)
    image = img_transform(image).unsqueeze(0)
    return image
def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
# mm = 0
for dataset in test_datasets:
    # mm = mm +1
    # print(mm)

    save_path = './results/LENS_Net/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/Image/'
    # print(dataset)
    gt_root = dataset_path + dataset + '/GT/'

    images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
    gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
    images = sorted(images)
    gts = sorted(gts)
    # print(images)
    i = 0
    for image, gt in zip(images, gts):
        print(i)
        i += 1
        filename = gt[-8:]

        # 1. 加载并预处理数据
        image = transform_image(image)  # [1, C, H, W]
        gt = binary_loader(gt)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.to(device)
        with torch.no_grad():
            # T = 4
            # image = image.unsqueeze(0).expand(T, -1, -1, -1, -1)  # [T, 1, C, H, W]
            res, _ , _ , _ = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()

        # 3. 后处理并保存
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        imageio.imsave(save_path + filename, res)
