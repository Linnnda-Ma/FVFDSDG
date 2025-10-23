from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from torch.cuda.amp import autocast
from torch.utils import data
from deeplearning.utils.file_utils import check_folders
from batchgenerators.utilities.file_and_folder_operations import *
from deeplearning.models.unet_fvfdsdg import UNetFVFDSDG
from deeplearning.datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from deeplearning.datasets.utils.convert_csv_to_list import convert_labeled_list
from deeplearning.datasets.utils.transform import collate_fn_ts
from deeplearning.utils.metrics.metrics import *
from deeplearning.utils.visualization import visualization_as_nii


def inference(chk_name, gpu, log_folder, patch_size, root_folder, ts_csv, inference_tag='all', tau=0.1):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    visualization_folder = join(visualization_folder, inference_tag)
    maybe_mkdir_p(visualization_folder)

    ts_img_list, ts_label_list = convert_labeled_list(ts_csv)
    if ts_label_list is None:
        evaluate = False
        ts_dataset = RIGA_unlabeled_set(root_folder, ts_img_list, patch_size)
    else:
        evaluate = True
        ts_dataset = RIGA_labeled_set(root_folder, ts_img_list, ts_label_list, patch_size)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=4,
                                                num_workers=2,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts)
    model = UNetFVFDSDG()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    assert isfile(join(model_folder, chk_name)), 'missing model checkpoint {}!'.format(join(model_folder, chk_name))
    params = torch.load(join(model_folder, chk_name))
    model.load_state_dict(params['model_state_dict'])

    seg_list = list()
    output_list = list()
    data_list = list()
    name_list = list()
    with torch.no_grad():
        model.eval()
        for iter, batch in enumerate(ts_dataloader):
            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            name = batch['name']
            with autocast():
                output = model(data, tau=tau)
            output_sigmoid = torch.sigmoid(output).cpu().numpy()
            seg_list.append(batch['seg'])
            output_list.append(output_sigmoid)
            data_list.append(batch['data'])
            name_list.append(name)
    all_data = list()
    all_seg = list()
    all_output = list()
    all_name = list()
    for i in range(len(data_list)):
        for j in range(data_list[i].shape[0]):
            all_data.append(data_list[i][j])
            all_seg.append(seg_list[i][j])
            all_output.append(output_list[i][j])
            all_name.append(name_list[i][j])
    all_data = np.stack(all_data)
    all_seg = np.stack(all_seg)
    all_output = np.stack(all_output)
#修改2
    visualization_as_nii(all_data[:, 0].astype(np.float32), join(visualization_folder, 'data_channel0.nii.gz'))
    visualization_as_nii(all_data[:, 1].astype(np.float32), join(visualization_folder, 'data_channel1.nii.gz'))
    visualization_as_nii(all_data[:, 2].astype(np.float32), join(visualization_folder, 'data_channel2.nii.gz'))
    visualization_as_nii(all_output[:, 0].astype(np.float32), join(visualization_folder, 'output_vessel.nii.gz'))
    # visualization_as_nii(all_output[:, 1].astype(np.float32), join(visualization_folder, 'output_cup.nii.gz'))

    if evaluate:
        visualization_as_nii(all_seg[:, 0].astype(np.float32), join(visualization_folder, 'seg.nii.gz'))
#修改3
                # 计算血管的 Dice 系数和 IOU
        vessel_dice, vessel_dice_list = get_hard_dice(
            torch.from_numpy(all_output[:, 0]),
            torch.from_numpy((all_seg[:, 0] > 0).astype(np.float32)),
            return_list=True
        )
        vessel_iou, vessel_iou_list = get_hard_iou(
            torch.from_numpy(all_output[:, 0]),
            torch.from_numpy((all_seg[:, 0] > 0).astype(np.float32)),
            return_list=True
        )
        vessel_hd95, vessel_hd95_list = get_hard_hd95(
            torch.from_numpy(all_output[:, 0]),
            torch.from_numpy((all_seg[:, 0] > 0).astype(np.float32)),
            return_list=True
        )
        vessel_ASSD, vessel_ASSD_list = get_hard_assd(
            torch.from_numpy(all_output[:, 0]),
            torch.from_numpy((all_seg[:, 0] > 0).astype(np.float32)),
            return_list=True
        )
        # 新增precision, recall, f1计算
        vessel_precision, vessel_precision_list = get_hard_precision(
            torch.from_numpy(all_output[:, 0]),
            torch.from_numpy((all_seg[:, 0] > 0).astype(np.float32)),
            return_list=True
        )
        vessel_recall, vessel_recall_list = get_hard_recall(
            torch.from_numpy(all_output[:, 0]),
            torch.from_numpy((all_seg[:, 0] > 0).astype(np.float32)),
            return_list=True
        )
        vessel_f1, vessel_f1_list = get_hard_f1(
            torch.from_numpy(all_output[:, 0]),
            torch.from_numpy((all_seg[:, 0] > 0).astype(np.float32)),
            return_list=True
        )
        
        metrics_str = 'Tag: {}\n  Vessel Dice: {}; Vessel IOU: {}; Vessel ASSD: {}; Vessel HD95: {};\n'.format(
            inference_tag, vessel_dice, vessel_iou, vessel_ASSD, vessel_hd95
        )
        # 新增precision, recall, f1的打印
        metrics_str += '  Vessel Precision: {}; Vessel Recall: {}; Vessel F1-score: {}'.format(
            vessel_precision, vessel_recall, vessel_f1
        )
        print(metrics_str)
        with open(join(metrics_folder, '{}.txt'.format(inference_tag)), 'w') as f:
            f.write(metrics_str)

        # 保存每张图像的评估结果到 CSV 文件（新增precision, recall, f1列）
        with open(join(metrics_folder, '{}.csv'.format(inference_tag)), 'w') as f:
            # 写入表头
            f.write('Name,Dice,IOU,ASSD,HD95,Precision,Recall,F1\n')
            for i in range(len(vessel_dice_list)):
                f.write('{},{},{},{},{},{},{},{}\n'.format(
                    all_name[i], 
                    vessel_dice_list[i], 
                    vessel_iou_list[i], 
                    vessel_ASSD_list[i], 
                    vessel_hd95_list[i],
                    vessel_precision_list[i],
                    vessel_recall_list[i],
                    vessel_f1_list[i]
                ))