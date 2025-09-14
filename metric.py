import os
import cv2
from tqdm import tqdm
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
# path = "datasets/test_dataset/EORSSD/GT/0004.png"
# mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
method = 'SggNet'
for _data_name in ['EORSSD']:#, 'COD10K', 'NC4K', 'CHAMELEON,'EORSSD‘]:
    print("eval-dataset: {}".format(_data_name))
    mask_root = 'datasets/test_dataset/ORSSD/GT/' # change path
    pred_root = 'results/SpikeNetv11/ORSSD/' # change path
    # pred_root = '/home/SNN-transformer/GeleNet-main/GeleNet_saliencymap_PVT/EORSSD'
    pred_name_list = sorted(os.listdir(pred_root))
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    for pred_name in tqdm(pred_name_list, total=len(pred_name_list)):
        mask_path = os.path.join(mask_root, pred_name)
        pred_path = os.path.join(pred_root, pred_name)
        # print(mask_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        # print(mask)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    results = {
        "Smeasure": sm,
        "wFmeasure": wfm,

        "maxFm": fm["curve"].max(),

        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),

        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "adpEm": em["adp"],

        "MAE": mae,
    }

    print(results)
    file = open("../results/eval_results.txt", "a")
    # file.write(method+' '+_data_name+' '+str(ckpt_save)+'\n')

print("Eval finished!")
