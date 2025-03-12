#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import math
import pprint
import torch.nn as nn 
import numpy as np
import torchvision
import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.models.contrastive import (
    contrastive_forward,
    contrastive_parameter_surgery,
)
from slowfast.utils.meters_cbm import AVAMeter, EpochTimer, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule
#from utils.DIPXv2 import CustomDataset
from utils.DIPX_random import CustomDataset
logger = logging.get_logger(__name__)



@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, train_loader, writer):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    #val_meter.iter_tic()

    for cur_iter, batch in enumerate(val_loader):
        
            
            batch_size = cfg.TEST.BATCH_SIZE
            *images,cls,gaze,ego = batch
            
            images = [img.cuda(non_blocking=True) for img in images]
            images = [img.type(torch.cuda.FloatTensor) for img in images]
            labels = cls.cuda(non_blocking=True)


            #val_meter.data_toc()

            if cfg.DETECTION.ENABLE:
                # Compute the predictions.
                preds = model(inputs, meta["boxes"])
                ori_boxes = meta["ori_boxes"]
                metadata = meta["metadata"]

                if cfg.NUM_GPUS:
                    preds = preds.cpu()
                    ori_boxes = ori_boxes.cpu()
                    metadata = metadata.cpu()

                if cfg.NUM_GPUS > 1:
                    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                    ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                    metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

                #val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(preds, ori_boxes, metadata)

            else:
                if cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                    if not cfg.CONTRASTIVE.KNN_ON:
                        return
                    train_labels = (
                        model.module.train_labels
                        if hasattr(model, "module")
                        else model.train_labels
                    )
                    yd, yi = model(inputs, index, time)
                    K = yi.shape[1]
                    C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
                    candidates = train_labels.view(1, -1).expand(batch_size, -1)
                    retrieval = torch.gather(candidates, 1, yi)
                    retrieval_one_hot = torch.zeros((batch_size * K, C)).cuda()
                    retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                    yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
                    probs = torch.mul(
                        retrieval_one_hot.view(batch_size, -1, C),
                        yd_transform.view(batch_size, -1, 1),
                    )
                    preds = torch.sum(probs, 1)
                else:
                    #inputs = images[0].unsqueeze(0)
                    preds = model(images[0],images[1])

                if cfg.DATA.MULTI_LABEL:
                    if cfg.NUM_GPUS > 1:
                        preds, labels = du.all_gather([preds, labels])
                else:
                    if cfg.DATA.IN22k_VAL_IN1K != "":
                        preds = preds[:, :1000]
                    # Compute the errors.
                    num_topks_correct = metrics.topks_correct(preds[0], labels, (1, 5))

                    # Combine the errors across the GPUs.
                    top1_err, top5_err = [
                        (1.0 - x / preds[0].size(0)) * 100.0 for x in num_topks_correct
                    ]
                    if cfg.NUM_GPUS > 1:
                        top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                    # Copy the errors from GPU to CPU (sync point).
                    top1_err, top5_err = top1_err.item(), top5_err.item()

                    #val_meter.iter_toc()
                    # Update and log stats.
                    val_meter.update_stats(
                        top1_err,
                        top5_err,
                        batch_size
                        * max(
                            cfg.NUM_GPUS, 1
                        ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                    )
                    # write to tensorboard format if available.
                    if writer is not None:
                        writer.add_scalars(
                            {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                            global_step=len(val_loader) * cur_epoch + cur_iter,
                        )
                
                val_meter.update_predictions(preds, labels, gaze, ego)
                #import pdb;pdb.set_trace()
            val_meter.log_iter_stats(cur_epoch, cur_iter)
    #val_meter.iter_tic()


    # Log epoch stats.
    
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars({"Val/mAP": val_meter.full_map}, global_step=cur_epoch)
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [label.clone().detach() for label in val_meter.all_labels]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(preds=all_preds, labels=all_labels, global_step=cur_epoch)

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        flops, params = misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(cfg, "train", is_precise_bn=True)
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    flops, params = 0.0, 0.0
    # if du.is_master_proc() and cfg.LOG_MODEL_INFO:
    #     flops, params = misc.log_model_info(model, cfg, use_train_input=True)
    #import pdb;pdb.set_trace()
    cu.load_test_checkpoint(cfg, model)
    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    start_epoch = 0


    train_csv = "/scratch/mukilv2/dipx/train.csv"
    val_csv = "/scratch/mukilv2/dipx/val.csv"
    transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #transform= None
    train_subset = CustomDataset(train_csv, rand_load=2, transform =transform )
    val_subset = CustomDataset(val_csv,rand_load=2, transform =transform)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=cfg.TRAIN.BATCH_SIZE//max(1,cfg.NUM_GPUS),pin_memory=True, shuffle= True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=cfg.TEST.BATCH_SIZE//max(1,cfg.NUM_GPUS))


    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)
    #import pdb;pdb.set_trace()
    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    #import pdb;pdb.set_trace()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cur_epoch > 0 and cfg.DATA.LOADER_CHUNK_SIZE > 0:
            num_chunks = math.ceil(
                cfg.DATA.LOADER_CHUNK_OVERALL_SIZE / cfg.DATA.LOADER_CHUNK_SIZE
            )
            skip_rows = (cur_epoch) % num_chunks * cfg.DATA.LOADER_CHUNK_SIZE
            logger.info(
                f"=================+++ num_chunks {num_chunks} skip_rows {skip_rows}"
            )
            cfg.DATA.SKIP_ROWS = skip_rows
            logger.info(f"|===========| skip_rows {skip_rows}")
            # train_loader = loader.construct_loader(cfg, "train")
            # loader.shuffle_dataset(train_loader, cur_epoch)

        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(
                        cfg.OUTPUT_DIR, task=cfg.TASK
                    )
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer)

        # Shuffle the dataset.
        #loader.shuffle_dataset(train_loader, cur_epoch)
        # if hasattr(train_loader.dataset, "_set_epoch_num"):
        #     train_loader.dataset._set_epoch_num(cur_epoch)
        # Train for one epoch.
        epoch_timer.epoch_tic()
        #import pdb;pdb.set_trace()

    eval_epoch(val_loader, model, val_meter, start_epoch, cfg, train_loader, writer)
    if writer is not None:
        writer.close()
    result_string = (
        "_p{:.2f}_f{:.2f} _t{:.2f}_m{:.2f} _a{:.2f} Top5 Acc: {:.2f} MEM: {:.2f} f: {:.4f}"
        "".format(
            params / 1e6,
            flops,
            (
                epoch_timer.median_epoch_time() / 60.0
                if len(epoch_timer.epoch_times)
                else 0.0
            ),
            misc.gpu_mem_usage(),
            100 - val_meter.min_top1_err,
            100 - val_meter.min_top5_err,
            misc.gpu_mem_usage(),
            flops,
        )
    )
    logger.info("training done: {}".format(result_string))

    return result_string
