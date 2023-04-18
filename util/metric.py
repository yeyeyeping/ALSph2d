from pymic.util.evaluation_seg import binary_dice, binary_iou, binary_assd


def get_metric(pred, gt):
    assert pred.ndim == 3 and gt.ndim == 3
    return binary_dice(pred, gt), binary_iou(pred, gt), binary_assd(pred, gt)
