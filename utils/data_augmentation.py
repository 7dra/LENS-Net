import os
from PIL import Image

# Set your paths
imPath = 'datasets/train_dataset/ORI-4199/image/'
GtPath = 'datasets/train_dataset/ORI-4199/GT/'

# Get all ground truth images
gt_images = [f for f in os.listdir(GtPath) if f.endswith('.png')]
imagesNum = len(gt_images)

for i in range(imagesNum):
    im_name = gt_images[i][:-4]  # Remove .png extension

    # Load images
    gt = Image.open(os.path.join(GtPath, gt_images[i]))
    im = Image.open(os.path.join(imPath, f'{im_name}.jpg'))

    # Rotate 90 degrees
    im_1 = im.rotate(90)
    gt_1 = gt.rotate(90)
    im_1.save(os.path.join(imPath, f'{im_name}_90.jpg'))
    gt_1.save(os.path.join(GtPath, f'{im_name}_90.png'))

    # Rotate 180 degrees
    im_2 = im.rotate(180)
    gt_2 = gt.rotate(180)
    im_2.save(os.path.join(imPath, f'{im_name}_180.jpg'))
    gt_2.save(os.path.join(GtPath, f'{im_name}_180.png'))

    # Rotate 270 degrees
    im_3 = im.rotate(270)
    gt_3 = gt.rotate(270)
    im_3.save(os.path.join(imPath, f'{im_name}_270.jpg'))
    gt_3.save(os.path.join(GtPath, f'{im_name}_270.png'))

    # Flip left-right
    fl_im = im.transpose(Image.FLIP_LEFT_RIGHT)
    fl_gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
    fl_im.save(os.path.join(imPath, f'{im_name}_fl.jpg'))
    fl_gt.save(os.path.join(GtPath, f'{im_name}_fl.png'))

    # Flip + rotate 90
    im_1 = fl_im.rotate(90)
    gt_1 = fl_gt.rotate(90)
    im_1.save(os.path.join(imPath, f'{im_name}_fl90.jpg'))
    gt_1.save(os.path.join(GtPath, f'{im_name}_fl90.png'))

    # Flip + rotate 180
    im_2 = fl_im.rotate(180)
    gt_2 = fl_gt.rotate(180)
    im_2.save(os.path.join(imPath, f'{im_name}_fl180.jpg'))
    gt_2.save(os.path.join(GtPath, f'{im_name}_fl180.png'))

    # Flip + rotate 270
    im_3 = fl_im.rotate(270)
    gt_3 = fl_gt.rotate(270)
    im_3.save(os.path.join(imPath, f'{im_name}_fl270.jpg'))
    gt_3.save(os.path.join(GtPath, f'{im_name}_fl270.png'))