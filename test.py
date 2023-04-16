from model import UNet
import torch
from dataset.SphDataset import Dataset3d
import numpy as np
from tqdm import tqdm
from util import label_smooth
from monai.networks.utils import one_hot
from monai.metrics import Cumulative, DiceMetric, MeanIoU, SurfaceDistanceMetric
from util.trainer import NoInfSurfaceDistanceMetric
from scipy.ndimage import zoom


def test(model: torch.nn.Module, dataloader, batch_size, input_size):
    device = next(iter(model.parameters())).device
    dice_metric, meaniou_metric, assd_metric = DiceMetric(include_background=False), MeanIoU(
        include_background=False), NoInfSurfaceDistanceMetric(include_background=False, symmetric=True)
    model.eval()
    tbar = tqdm(dataloader)
    dice_his, mean_iou_his, assd_hist = [], [], []
    for idx, (img, mask) in enumerate(tbar):
        pred_volume = np.empty((0, img.shape[-2], img.shape[-1]), dtype=np.float32)
        img, mask = img[0], mask[0]
        h, w = img.shape[-2], img.shape[-1]
        for batch in range(0, img.shape[0], batch_size):
            last = batch + batch_size
            batch_slices = img[batch:] if last >= img.shape[0] else img[batch:last]

            batch_slices = zoom(batch_slices, (1, 1, input_size / h, input_size / w), order=0,
                                mode='nearest')
            batch_slices = torch.from_numpy(batch_slices).to(device)
            output = model(batch_slices)
            if isinstance(output, tuple):
                output = output[0]
            batch_pred_mask = output.argmax(dim=1).cpu()
            batch_pred_mask = zoom(batch_pred_mask, (1, h / input_size, w / input_size), order=0,
                                   mode='nearest')
            pred_volume = np.concatenate([pred_volume, batch_pred_mask])
            torch.cuda.empty_cache()

        mask_onehot = one_hot(mask, 2)
        volume_pred_mask = torch.from_numpy(label_smooth(pred_volume)).unsqueeze(1)
        dice = dice_metric(y_pred=volume_pred_mask, y=mask_onehot)
        iou = meaniou_metric(y_pred=volume_pred_mask, y=mask_onehot)
        assd = assd_metric(y_pred=volume_pred_mask, y=mask_onehot)
        dice, iou = dice[dice.isnan() == 0].mean(), iou[iou.isnan() == 0].mean()
        assd = assd[(assd.isnan() == 0) & assd.isfinite()].mean()
        tbar.set_description(
            f"Test | Dice:{dice:.3f} Mean IoU: {iou:.2f} asd: {assd:.2f} ")
        dice_his.append(dice)
        mean_iou_his.append(iou)
        assd_hist.append(assd)
    return np.array(dice_his), np.array(mean_iou_his), np.array(assd_hist)


device = "cuda"
model = UNet(1, 2, 16).to(device)




model.load_state_dict(torch.load("/home/yeep/桌面/cycle=4&dice=0.844&time=1681041803.2627594.pth"))

dataset_val = Dataset3d(folder="data/val")


dataloader = torch.utils.data.DataLoader(dataset_val,
                                         batch_size=1,
                                         persistent_workers=True,
                                         pin_memory=True,
                                         prefetch_factor=4,
                                         num_workers=4)

dice_his, mean_iou_his, assd_hist = test(model, dataloader, 16, 416)


