from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from torch.cuda.amp import autocast
from torch.utils import data
from deeplearning.utils.file_utils import check_folders
from batchgenerators.utilities.file_and_folder_operations import *
from deeplearning.models.unet import UNet
from deeplearning.datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from deeplearning.datasets.utils.convert_csv_to_list import convert_labeled_list
from deeplearning.datasets.utils.transform import collate_fn_ts
from deeplearning.utils.metrics.metrics import *
from deeplearning.utils.visualization import visualization_as_nii


def inference(chk_name, gpu, log_folder, patch_size, root_folder, ts_csv, inference_tag='all'):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    visualization_folder = join(visualization_folder, inference_tag)
    maybe_mkdir_p(visualization_folder)

    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=1)
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
    model = UNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    assert isfile(join(model_folder, chk_name)), 'missing model checkpoint {}!'.format(join(model_folder, chk_name))
    params = torch.load(join(model_folder, chk_name))
    model.load_state_dict(params['model_state_dict'])

    seg_list = []
    output_list = []
    data_list = []
    name_list = []
    with torch.no_grad():
        model.eval()
        for iter, batch in enumerate(ts_dataloader):
            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)  # Retinal segmentation only
            with autocast():
                output = model(data)

            output_sigmoid = torch.sigmoid(output).cpu().numpy()
            seg_list.append(batch['seg'])
            output_list.append(output_sigmoid)
            data_list.append(batch['data'])
            name_list.append(batch['name'])

    all_data = []
    all_seg = []
    all_output = []
    all_name = []
    for i in range(len(data_list)):
        for j in range(data_list[i].shape[0]):
            all_data.append(data_list[i][j])
            all_seg.append(seg_list[i][j])
            all_output.append(output_list[i][j])
            all_name.append(name_list[i][j])
    
    all_data = np.stack(all_data)
    all_seg = np.stack(all_seg)
    all_output = np.stack(all_output)
    
    # Visualizations for retinal data only
    visualization_as_nii(all_data[:, 0].astype(np.float32), join(visualization_folder, 'data_channel0.nii.gz'))
    visualization_as_nii(all_data[:, 1].astype(np.float32), join(visualization_folder, 'data_channel1.nii.gz'))
    visualization_as_nii(all_data[:, 2].astype(np.float32), join(visualization_folder, 'data_channel2.nii.gz'))
    visualization_as_nii(all_output[:, 0].astype(np.float32), join(visualization_folder, 'output_retina.nii.gz'))

    if evaluate:
        visualization_as_nii(all_seg[:, 0].astype(np.float32), join(visualization_folder, 'seg.nii.gz'))
        
        # Compute metrics
        retinal_dice, retinal_dice_list = get_hard_dice(torch.from_numpy(all_output[:, 0]), torch.from_numpy((all_seg[:, 0] > 0) * 1.0), return_list=True)
        retinal_iou, retinal_iou_list = get_hard_iou(torch.from_numpy(all_output[:, 0]), torch.from_numpy((all_seg[:, 0] > 0) * 1.0), return_list=True)
        retinal_hd95, retinal_hd95_list = get_hard_hd95(torch.from_numpy(all_output[:, 0]), torch.from_numpy((all_seg[:, 0] > 0) * 1.0), return_list=True)
        retinal_precision, retinal_precision_list = get_hard_precision(torch.from_numpy(all_output[:, 0]), torch.from_numpy((all_seg[:, 0] > 0) * 1.0), return_list=True)
        retinal_recall, retinal_recall_list = get_hard_recall(torch.from_numpy(all_output[:, 0]), torch.from_numpy((all_seg[:, 0] > 0) * 1.0), return_list=True)
        retinal_f1, retinal_f1_list = get_hard_f1(torch.from_numpy(all_output[:, 0]), torch.from_numpy((all_seg[:, 0] > 0) * 1.0), return_list=True)

        # Calculate means and standard deviations
        metrics_str = (
            f'Tag: {inference_tag}\n'
            f'  Retina Dice: {np.mean(retinal_dice_list):.4f} ± {np.std(retinal_dice_list):.4f}\n'
            f'  Retina IOU: {np.mean(retinal_iou_list):.4f} ± {np.std(retinal_iou_list):.4f}\n'
            f'  Retina HD95: {np.mean(retinal_hd95_list):.4f} ± {np.std(retinal_hd95_list):.4f}\n'
            f'  Retina Precision: {np.mean(retinal_precision_list):.4f} ± {np.std(retinal_precision_list):.4f}\n'
            f'  Retina Recall: {np.mean(retinal_recall_list):.4f} ± {np.std(retinal_recall_list):.4f}\n'
            f'  Retina F1: {np.mean(retinal_f1_list):.4f} ± {np.std(retinal_f1_list):.4f}'
        )
        print(metrics_str)

        # Save metrics to text and CSV files
        with open(join(metrics_folder, f'{inference_tag}.txt'), 'w') as f:
            f.write(metrics_str)

        with open(join(metrics_folder, f'{inference_tag}.csv'), 'w') as f:
            f.write("Image Name, Retina Dice, Retina IOU, Retina HD95, Retina Precision, Retina Recall, Retina F1\n")
            for i in range(len(retinal_dice_list)):
                f.write(f'{all_name[i]},{retinal_dice_list[i]},{retinal_iou_list[i]},{retinal_hd95_list[i]},'
                        f'{retinal_precision_list[i]},{retinal_recall_list[i]},{retinal_f1_list[i]}\n')
        