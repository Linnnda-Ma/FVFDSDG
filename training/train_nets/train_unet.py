from time import time
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data
import torch.nn as nn
from deeplearning.utils.file_utils import check_folders
from batchgenerators.utilities.file_and_folder_operations import *
from deeplearning.models.unet import UNet
from deeplearning.datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set
from deeplearning.datasets.utils.convert_csv_to_list import convert_labeled_list
from deeplearning.datasets.utils.transform import collate_fn_tr_only_sd_trans, collate_fn_ts
from deeplearning.utils.lr import adjust_learning_rate
from deeplearning.utils.metrics.metrics import *
from torchvision.utils import make_grid

def train(args):
    model_name = args.model
    gpu = tuple(args.gpu)
    log_folder = args.log_folder
    tag = args.tag
    log_folder = join(log_folder, model_name+'_'+tag)
    patch_size = tuple(args.patch_size)
    batch_size = args.batch_size
    initial_lr = args.initial_lr
    save_interval = args.save_interval
    num_epochs = args.num_epochs
    continue_training = args.continue_training
    num_threads = args.num_threads
    root_folder = args.root
    tr_csv = tuple(args.tr_csv)
    ts_csv = tuple(args.ts_csv)
    shuffle = not args.no_shuffle

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    writer = SummaryWriter(log_dir=tensorboard_folder)

    tr_img_list, tr_label_list = convert_labeled_list(tr_csv)
    tr_dataset = RIGA_labeled_set(root_folder, tr_img_list, tr_label_list, patch_size)
    ts_img_list, ts_label_list = convert_labeled_list(ts_csv)
    ts_dataset = RIGA_labeled_set(root_folder, ts_img_list, ts_label_list, patch_size)
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=collate_fn_tr_only_sd_trans)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads//2,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts)

    model = UNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)

    start_epoch = 0
    if continue_training:
        assert isfile(join(model_folder, 'model_latest.model')), 'missing model checkpoint!'
        params = torch.load(join(model_folder, 'model_latest.model'))
        model.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        start_epoch = params['epoch']
    print('start epoch: {}'.format(start_epoch))

    amp_grad_scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    start = time()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}:'.format(epoch))
        start_epoch = time()
        model.train()
        
        lr = adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs)
        print('  lr: {}'.format(lr))

        train_loss_list = []
        train_dice_list = []
        train_IOU_list = []

        for iter, batch in enumerate(tr_dataloader):
            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
            optimizer.zero_grad()
            with autocast():
                output = model(data)
                loss = criterion(output[:, 0], seg[:, 0])  # Loss calculation for retina

            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
            train_loss_list.append(loss.detach().cpu().numpy())

            output_sigmoid = torch.sigmoid(output)
            train_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), seg[:, 0].cpu()))
            train_IOU_list.append(get_hard_iou(output_sigmoid[:, 0].cpu(), seg[:, 0].cpu()))

        mean_tr_loss = np.mean(train_loss_list)
        mean_tr_dice = np.mean(train_dice_list)
        mean_tr_IOU = np.mean(train_IOU_list)
        
        writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        writer.add_scalar("Train Scalars/Dice", mean_tr_dice, epoch)
        writer.add_scalar("Train Scalars/IOU", mean_tr_IOU, epoch)
        
        print('  Tr loss: {}\n'
              '  Tr dice: {};\n'
              '  Tr IOU: {};'.format(mean_tr_loss, mean_tr_dice, mean_tr_IOU))

        val_loss_list = []
        val_dice_list = []
        val_IOU_list = []
        
        with torch.no_grad():
            model.eval()
            for iter, batch in enumerate(ts_dataloader):
                data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                with autocast():
                    output = model(data)
                    loss = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0)  # Only consider retina
                val_loss_list.append(loss.detach().cpu().numpy())
                output_sigmoid = torch.sigmoid(output)
                val_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), seg[:, 0].cpu()))
                val_IOU_list.append(get_hard_iou(output_sigmoid[:, 0].cpu(), seg[:, 0].cpu()))
        
        # Calculate mean and standard deviation for Dice and IoU
        mean_val_loss = np.mean(val_loss_list)
        mean_val_dice = np.mean(val_dice_list)
        mean_val_IOU = np.mean(val_IOU_list)
        
        std_val_dice = np.std(val_dice_list)  # Standard deviation of Dice
        std_val_IOU = np.std(val_IOU_list)  # Standard deviation of IoU
        
        # Log results to TensorBoard
        writer.add_scalar("Val Scalars/Val Loss", mean_val_loss, epoch)
        writer.add_scalar("Val Scalars/Dice", mean_val_dice, epoch)
        writer.add_scalar("Val Scalars/IOU", mean_val_IOU, epoch)
        writer.add_scalar("Val Scalars/Dice STD", std_val_dice, epoch)  # Log standard deviation of Dice
        writer.add_scalar("Val Scalars/IOU STD", std_val_IOU, epoch)  # Log standard deviation of IoU
        
        # Optionally, log images
        writer.add_image('Val/Input', make_grid(data[:10], 10, normalize=True), epoch)
        writer.add_image('Val/Output', make_grid(output_sigmoid[:10, 0][:, np.newaxis], 10, normalize=True), epoch)
        writer.add_image('Val/Seg', make_grid(seg[:10], 10, normalize=True), epoch)
        
        # Print the results including standard deviations
        print(f'  Val loss: {mean_val_loss}\n'
              f'  Val dice: {mean_val_dice} ± {std_val_dice};\n'
              f'  Val IOU: {mean_val_IOU} ± {std_val_IOU};')


        if epoch % save_interval == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format('latest'))
            torch.save(saved_model, join(model_folder, 'model_latest.model'))

        if (epoch+1) % 200 == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format(epoch+1))
            torch.save(saved_model, join(model_folder, 'model_{}.model'.format(epoch+1)))

        time_per_epoch = time() - start_epoch
        print('  Durations: {}'.format(time_per_epoch))
        writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)

    saved_model = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    print('Saving model_{}.model...'.format('final'))
    torch.save(saved_model, join(model_folder, 'model_final.model'))
    if isfile(join(model_folder, 'model_latest.model')):
        os.remove(join(model_folder, 'model_latest.model'))

    total_time = time() - start
    print("Running %d epochs took a total of %.2f seconds." % (num_epochs, total_time))

    # Inference
    from deeplearning.inference.inference_nets.inference_unet import inference
    for ts_csv_path in ts_csv:
        inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
        print("Running inference: {}".format(inference_tag))
        inference('model_final.model', gpu, log_folder, patch_size, root_folder, [ts_csv_path], inference_tag)
