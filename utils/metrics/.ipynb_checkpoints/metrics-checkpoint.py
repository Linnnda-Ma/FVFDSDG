from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
import numpy as np
import torch
from scipy.stats import ttest_rel  # 用于p-value计算

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    zero = torch.zeros_like(result)
    one = torch.ones_like(result)
    result = torch.where(result > 0.5, one, zero)
    reference = torch.where(reference > 0.5, one, zero)
    
    result = result.numpy()
    reference = reference.numpy()
    result = result.astype(int)
    reference = reference.astype(int)

    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    footprint = generate_binary_structure(result.ndim, connectivity)

    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def get_assd(result, reference, voxelspacing=None, connectivity=1):
    sds1 = __surface_distances(result, reference, voxelspacing, connectivity)
    sds2 = __surface_distances(reference, result, voxelspacing, connectivity)
    all_distances = np.hstack((sds1, sds2))
    assd = np.mean(all_distances)
    return assd

def get_hard_assd(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    assd_list = []
    
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        assd = get_assd(output, mask)
        assd_list.append(assd)
    
    if return_list:
        return np.mean(assd_list), assd_list
    else:
        return np.mean(assd_list)

def hd95(result, reference, voxelspacing=None, connectivity=1):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95

def get_hard_hd95(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    hd95_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        hd95_list.append(hd95(output, mask))
    if return_list:
        return np.mean(hd95_list), hd95_list
    else:
        return np.mean(hd95_list)

def get_dice_threshold(output, mask, threshold=0.5):
    smooth = 1e-6
    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    intersection = (output * mask).sum()
    dice = (2. * intersection + smooth) / (output.sum() + mask.sum() + smooth)
    return dice

def get_hard_dice(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    dice_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        dice_list.append(get_dice_threshold(output, mask, threshold=0.5))
    if return_list:
        return np.mean(dice_list), dice_list
    else:
        return np.mean(dice_list)

def get_iou_threshold(output, mask, threshold=0.5):
    smooth = 1e-6
    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    intersection = (mask * output).sum()
    union = (mask + output).sum() - intersection
    iou = intersection / union if union > 0 else 0.0
    return iou

def get_hard_iou(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    iou_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        iou_list.append(get_iou_threshold(output, mask, threshold=0.5))
    if return_list:
        return np.mean(iou_list), iou_list
    else:
        return np.mean(iou_list)

# 新增部分：precision, recall, F1-score计算
def get_precision(output, mask, threshold=0.5):
    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    tp = (output * mask).sum().item()
    fp = ((output == 1) * (mask == 0)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision

def get_recall(output, mask, threshold=0.5):
    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    tp = (output * mask).sum().item()
    fn = ((output == 0) * (mask == 1)).sum().item()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall

def get_f1_score(output, mask, threshold=0.5):
    precision = get_precision(output, mask, threshold)
    recall = get_recall(output, mask, threshold)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def get_hard_precision(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    precision_list = []
    for i in range(outputs.size(0)):
        precision_list.append(get_precision(outputs[i], masks[i]))
    if return_list:
        return np.mean(precision_list), precision_list
    else:
        return np.mean(precision_list)

def get_hard_recall(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    recall_list = []
    for i in range(outputs.size(0)):
        recall_list.append(get_recall(outputs[i], masks[i]))
    if return_list:
        return np.mean(recall_list), recall_list
    else:
        return np.mean(recall_list)

def get_hard_f1(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    f1_list = []
    for i in range(outputs.size(0)):
        f1_list.append(get_f1_score(outputs[i], masks[i]))
    if return_list:
        return np.mean(f1_list), f1_list
    else:
        return np.mean(f1_list)

# 新增p-value计算，示例为配对t检验
def get_p_value(metric_list1, metric_list2):
    """
    计算两个指标列表的配对t检验p-value，用于比较两组结果的显著性差异
    :param metric_list1: 第一组指标值列表，数组或list
    :param metric_list2: 第二组指标值列表，数组或list
    :return: p-value
    """
    metric_list1 = np.array(metric_list1)
    metric_list2 = np.array(metric_list2)
    t_stat, p_value = ttest_rel(metric_list1, metric_list2)
    return p_value