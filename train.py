#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import torch
from torch.utils.data import DataLoader
import json
import numpy as np

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
from data.dataset import (
    PDBBindBenchmark, MixDatasetWrapper, DynamicBatchWrapper,
    BalancedDynamicBatchWrapper, PretrainBalancedDynamicBatchWrapper,
    LabelledPDBDataset, MultiClassLabelledPDBDataset,
    ProtInterfaceDataset
)
from data.distributed_sampler import DistributedSamplerResume
import models
import trainers
from utils.nn_utils import count_parameters
from data.pdb_utils import VOCAB

import wandb

def parse():
    parser = argparse.ArgumentParser(description='training')
    # data
    parser.add_argument('--train_set_path', type=str, default='LP_PDBBind/LP_PDBBind_train', help='path to train set')
    parser.add_argument('--valid_set_path', type=str, default='LP_PDBBind/LP_PDBBind_test',help='path to valid set')
    parser.add_argument('--task', type=str,  default='PDBBind',
                        choices=['pretrain_torsion', 'pretrain_torsion_masking', 'pretrain_gaussian',
                                 'binary_classifier', 'multiclass_classifier', 'masking', 
                                 'PDBBind', 'prot_interface', 'regression'])


    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=None, help='final learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of epochs where validation loss is not used for early stopping')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-5, help='linear learning rate warmup start lr')
    parser.add_argument('--warmup_end_lr', type=float, default=1e-3, help='linear learning rate warmup end lr')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--max_epoch', type=int, default=100, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients with too big norm')
    parser.add_argument('--save_dir', type=str, default='model_checkpoints', help='directory to save model and logs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--valid_batch_size', type=int, default=16, help='batch size of validation, default set to the same as training batch size')
    parser.add_argument('--max_n_vertex_per_gpu', type=int, default=None, help='if specified, ignore batch_size and form batch with dynamic size constrained by the total number of vertexes')
    parser.add_argument('--max_n_vertex_per_item', type=int, default=None, help='if max_n_vertex_per_gpu is specified, larger items will be randomly cropped')
    parser.add_argument('--valid_max_n_vertex_per_gpu', type=int, default=None, help='form batch with dynamic size constrained by the total number of vertexes')
    parser.add_argument('--balanced_sampler', action='store_true', default=False, help='use balanced sampler')
    parser.add_argument('--patience', type=int, default=-1, help='patience before early stopping')
    parser.add_argument('--save_topk', type=int, default=-1, help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--cycle_steps', type=int, default=100000, help='number of steps per cycle in lr_scheduler.CosineAnnealingWarmRestarts')

    # device
    parser.add_argument('--gpus', type=list, nargs='+', default=[0], help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    # model
    parser.add_argument('--atom_hidden_size', type=int, default=128, help='dimension of hidden states')
    parser.add_argument('--block_hidden_size', type=int, default=128, help='dimension of hidden states for blocks')
    parser.add_argument('--edge_size', type=int, default=16, help='Dimension of edge embeddings')
    parser.add_argument('--k_neighbors', type=int, default=8, help='Number of neighbors in KNN graph')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--bottom_global_message_passing', action="store_true", default=False, help='message passing between global nodes and normal nodes at the bottom level')
    parser.add_argument('--global_message_passing', action="store_true", default=False, help='message passing between global nodes and normal nodes at the top level')
    parser.add_argument('--fragmentation_method', type=str, default=None, choices=['PS_300'], help='fragmentation method for small molecules')

    # for pretraining
    parser.add_argument('--atom_noise', type=float, default=0, help='apply noise to atom coordinates')
    parser.add_argument('--translation_noise', type=float, default=0, help='apply global translation noise')
    parser.add_argument('--rotation_noise', type=float, default=0, help='apply global rotation noise')
    parser.add_argument('--torsion_noise', type=float, default=0, help='max torsion rotation noise')
    parser.add_argument('--max_rotation', type=float, default=np.pi/4, help='max global rotation angle')
    parser.add_argument('--tr_weight', type=float, default=1.0, help='Weight of translation loss')
    parser.add_argument('--rot_weight', type=float, default=1.0, help='Weight of rotation loss')
    parser.add_argument('--tor_weight', type=float, default=1.0, help='Weight of torsional loss')
    parser.add_argument('--atom_weight', type=float, default=1.0, help='Weight of atom loss')
    parser.add_argument('--mask_proportion', type=float, default=0, help='block masking rate')
    parser.add_argument('--mask_weight', type=float, default=1.0, help='block masking rate')
    parser.add_argument('--noisy_nodes_weight', type=float, default=0, help='coefficient for denoising loss during finetuning')
    parser.add_argument('--modality_embedding', action="store_true", default=False, help='add embedding for each modality')

    # load pretrain
    parser.add_argument('--pretrain_ckpt', type=str, default=None, help='path of the pretrained ckpt to load')
    parser.add_argument('--pretrain_config', type=str, default='/blue/yanjun.li/share/GatorMol/GatorMol/benchmark/ATOMICA/ckpt/pretrain_model_config.json', help='path of the pretrained config to load')
    parser.add_argument('--pretrain_weights', type=str, default='/blue/yanjun.li/share/GatorMol/GatorMol/benchmark/ATOMICA/ckpt/pretrain_model_weights.pt', help='path of the pretrained weights to load')
    parser.add_argument('--pretrain_state', type=str, default=None, help='path of the pretrained training state to load for resuming training')
    parser.add_argument('--partial_finetune', action="store_true", default=False, help='only finetune energy head')

    # for prediction
    parser.add_argument('--pred_dropout', type=float, default=0.0, help='dropout rate for prediction')
    parser.add_argument('--pred_nonlinearity', type=str, default='relu', choices=['relu', 'gelu', 'elu'], help='nonlinearity for prediction')
    parser.add_argument('--num_pred_layers', type=int, default=3, help='number of layers for prediction')
    parser.add_argument('--pred_hidden_size', type=int, default=32, help='hidden size for prediction')
    parser.add_argument('--num_projector_layers', type=int, default=3, help='number of layers for projector')
    parser.add_argument('--projector_hidden_size', type=int, default=256, help='hidden size for projector')
    parser.add_argument('--projector_dropout', type=float, default=0.0, help='dropout rate for projector')
    parser.add_argument('--block_embedding_size', type=int, default=None, help='embedding size for blocks')
    parser.add_argument('--block_embedding0_size', type=int, default=None, help='embedding size for blocks in segment0, block_embedding_size1 will be used for blocks in segment1')
    parser.add_argument('--block_embedding1_size', type=int, default=None, help='embedding size for blocks in segment1, block_embedding_size0 will be used for blocks in segment0')

    # logging
    parser.add_argument('--use_wandb', action="store_true", default=False, help='log to Weights and Biases')
    parser.add_argument('--use_raytune', action="store_true", default=False, help='log to RayTune')
    parser.add_argument('--run_name', type=str, default="test", help='model run name for logging')

    return parser.parse_args()


def create_dataset(task, path, path2=None, path3=None, fragment=None):    
    if task == 'pretrain_torsion':
        from data.dataset_pretrain import PretrainTorsionDataset
        dataset1 = PretrainTorsionDataset(path)
        print_log(f'Pretrain dataset {path} size: {len(dataset1)}')
        if path2 is None and path3 is None:
            return dataset1
        datasets = [dataset1]
        if path2 is not None:
            dataset2 = PretrainTorsionDataset(path2)
            datasets.append(dataset2)
            print_log(f'Pretrain dataset {path2} size: {len(dataset2)}')
        if path3 is not None:
            dataset3 = PretrainTorsionDataset(path3)
            datasets.append(dataset3)
            print_log(f'Pretrain dataset {path3} size: {len(dataset3)}')
        dataset = MixDatasetWrapper(*datasets)
        print_log(f'Mixed pretrain dataset size: {len(dataset)}')
    elif task == 'pretrain_torsion_masking':
        from data.dataset_pretrain import PretrainMaskedTorsionDataset
        dataset_args = {
            "mask_proportion": 0,
            "mask_token": VOCAB.symbol_to_idx(VOCAB.MASK),
            "vocab_to_mask": [VOCAB.symbol_to_idx(x[0]) for x in VOCAB.aas + VOCAB.bases + VOCAB.sms + VOCAB.frags],
            "atom_mask_token": VOCAB.get_atom_mask_idx(),
        }
        dataset1 = PretrainMaskedTorsionDataset(path, **dataset_args)
        print_log(f'Pretrain dataset {path} size: {len(dataset1)}')
        if path2 is None and path3 is None:
            return dataset1
        datasets = [dataset1]
        if path2 is not None:
            dataset2 = PretrainMaskedTorsionDataset(path2, **dataset_args)
            datasets.append(dataset2)
            print_log(f'Pretrain dataset {path2} size: {len(dataset2)}')
        if path3 is not None:
            dataset3 = PretrainMaskedTorsionDataset(path3, **dataset_args)
            datasets.append(dataset3)
            print_log(f'Pretrain dataset {path3} size: {len(dataset3)}')
        dataset = MixDatasetWrapper(*datasets)
        print_log(f'Mixed pretrain dataset size: {len(dataset)}')
    elif task == 'pretrain_gaussian':
        from data.dataset_pretrain import PretrainAtomDataset
        dataset1 = PretrainAtomDataset(path)
        print_log(f'Pretrain dataset {path} size: {len(dataset1)}')
        if path2 is None and path3 is None:
            return dataset1
        datasets = [dataset1]
        if path2 is not None:
            dataset2 = PretrainAtomDataset(path2)
            datasets.append(dataset2)
            print_log(f'Pretrain dataset {path2} size: {len(dataset2)}')
        if path3 is not None:
            dataset3 = PretrainAtomDataset(path3)
            datasets.append(dataset3)
            print_log(f'Pretrain dataset {path3} size: {len(dataset3)}')
        dataset = MixDatasetWrapper(*datasets)
        print_log(f'Mixed pretrain dataset size: {len(dataset)}')
    elif task == 'binary_classifier' or task == 'regression':
        dataset = LabelledPDBDataset(path)
        datasets = [dataset]
        if path2 is not None:
            dataset2 = LabelledPDBDataset(path2)
            datasets.append(dataset2)
        if path3 is not None:
            dataset3 = LabelledPDBDataset(path3)
            datasets.append(dataset3)
        if len(datasets) > 1:
            dataset = MixDatasetWrapper(*datasets)
    elif task == 'multiclass_classifier':
        dataset = MultiClassLabelledPDBDataset(path)
        datasets = [dataset]
        if path2 is not None:
            dataset2 = MultiClassLabelledPDBDataset(path2)
            datasets.append(dataset2)
        if path3 is not None:
            dataset3 = MultiClassLabelledPDBDataset(path3)
            datasets.append(dataset3)
        if len(datasets) > 1:
            dataset = MixDatasetWrapper(*datasets)
    elif task == "masking":
        from data.dataset_pretrain import PretrainMaskedDataset
        dataset_args = {
            "mask_proportion": 0,
            "mask_token": VOCAB.symbol_to_idx(VOCAB.MASK),
            "vocab_to_mask": [VOCAB.symbol_to_idx(x[0]) for x in VOCAB.aas + VOCAB.bases + VOCAB.sms + VOCAB.frags],
            "atom_mask_token": VOCAB.get_atom_mask_idx(),
        }
        dataset = PretrainMaskedDataset(path, **dataset_args)
        datasets = [dataset]
        if path2 is not None:
            dataset2 = PretrainMaskedDataset(path2, **dataset_args)
            datasets.append(dataset2)
        if path3 is not None:
            dataset3 = PretrainMaskedDataset(path3, **dataset_args)
            datasets.append(dataset3)
        if len(datasets) > 1:
            dataset = MixDatasetWrapper(*datasets)
    elif task == 'PDBBind':
        dataset = PDBBindBenchmark(path)
        if path2 is not None or path3 is not None:
            raise NotImplementedError('ProtInterfaceDataset does not support multiple datasets')
    elif task == "prot_interface":
        dataset = ProtInterfaceDataset(path)
        if path2 is not None or path3 is not None:
            raise NotImplementedError('ProtInterfaceDataset does not support multiple datasets')
    else:
        raise NotImplementedError(f'Dataset for {task} not implemented!')
    return dataset


def set_noise(dataset, args):
    from data.dataset_pretrain import PretrainAtomDataset, PretrainTorsionDataset, PretrainMaskedDataset, PretrainMaskedTorsionDataset
    if type(dataset) in [PretrainAtomDataset, PretrainTorsionDataset, PretrainMaskedTorsionDataset]:
        if args.atom_noise != 0 and args.torsion_noise != 0:
            raise ValueError('Cannot set both atom and torsion noise at the same time')
        if type(dataset) == PretrainAtomDataset and args.atom_noise != 0:
            dataset.set_atom_noise(args.atom_noise)
        if args.translation_noise != 0:
            dataset.set_translation_noise(args.translation_noise)
        if args.rotation_noise != 0:
            dataset.set_rotation_noise(args.rotation_noise, args.max_rotation)
        if args.max_n_vertex_per_item is not None:
            dataset.set_crop(args.max_n_vertex_per_item, args.fragmentation_method)
        if type(dataset) in [PretrainTorsionDataset, PretrainMaskedTorsionDataset] and args.torsion_noise != 0:
            dataset.set_torsion_noise(args.torsion_noise)
        if type(dataset) == PretrainMaskedTorsionDataset:
            dataset.mask_proportion = args.mask_proportion
    elif type(dataset) == PretrainMaskedDataset:
        dataset.mask_proportion = args.mask_proportion
    elif type(dataset) == MixDatasetWrapper:
        new_datasets = []
        for d in dataset.datasets:
            d = set_noise(d, args)
            new_datasets.append(d)
        dataset = MixDatasetWrapper(*new_datasets) # update the mix dataset wrapper with new dataset lengths
    return dataset


def create_trainer(model, train_loader, valid_loader, config, resume_state=None):
    model_type = type(model)
    if model_type in [models.AffinityPredictor, models.RegressionPredictor]:
        trainer = trainers.AffinityTrainer(model, train_loader, valid_loader, config)
    elif model_type == models.ClassifierModel:
        trainer = trainers.ClassifierTrainer(model, train_loader, valid_loader, config)
    elif model_type == models.MultiClassClassifierModel:
        trainer = trainers.MultiClassClassifierTrainer(model, train_loader, valid_loader, config)
    elif model_type == models.DenoisePretrainModel:
        if model.masking_objective:
            trainer = trainers.PretrainMaskingNoisingTrainer(
                model, train_loader, valid_loader, config, 
                resume_state=resume_state,  
            )
        else:
            trainer = trainers.PretrainTrainer(
                model, train_loader, valid_loader, config, 
                resume_state=resume_state,
            )
    elif model_type == models.DenoisePretrainModelWithBlockEmbedding:
        trainer = trainers.PretrainMaskingNoisingTrainerWithBlockEmbedding(
            model, train_loader, valid_loader, config, 
            resume_state=resume_state,
        )
    elif model_type == models.MaskedNodeModel:
        trainer = trainers.MaskingTrainer(model, train_loader, valid_loader, config)
    elif model_type == models.ProteinInterfaceModel:
        trainer = trainers.ProtInterfaceTrainer(model, train_loader, valid_loader, config)
    else:
        raise NotImplementedError(f'Trainer for model type {model_type} not implemented!')
    return trainer


def main(args):
    setup_seed(args.seed)
    VOCAB.load_tokenizer(args.fragmentation_method)
    # torch.autograd.set_detect_anomaly(True)
    if args.task == "pretrain_torsion_masking" or args.task == "masking":
        args.num_nodes = len(VOCAB.aas + VOCAB.bases + VOCAB.sms + VOCAB.frags)
    else:
        args.num_nodes = None
    model = models.create_model(args)

    ########### load your train / valid set ###########
    if args.task == 'PLA_noisy_nodes':
        train_task = 'PLA_noisy_nodes_train'
    else:
        train_task = args.task
    train_set = create_dataset(train_task, args.train_set_path,args.fragmentation_method)
    if args.valid_set_path is not None:
        valid_set = create_dataset(train_task, args.valid_set_path,args.fragmentation_method)
        print_log(f'Train: {len(train_set)}, validation: {len(valid_set)}')
    else:
        valid_set = None
        print_log(f'Train: {len(train_set)}, no validation')
    ########## define your model/trainer/trainconfig #########
    step_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    config = trainers.TrainConfig(
        args.save_dir, args.lr, args.max_epoch,
        cycle_steps=args.cycle_steps,
        warmup_epochs=args.warmup_epochs,
        warmup_start_lr=args.warmup_start_lr,
        warmup_end_lr=args.warmup_end_lr,
        patience=args.patience,
        grad_clip=args.grad_clip,
        save_topk=args.save_topk,
    )
    config.add_parameter(step_per_epoch=step_per_epoch,
                         final_lr=args.final_lr if args.final_lr is not None else args.lr)
    if args.valid_batch_size is None:
        args.valid_batch_size = args.batch_size

    if type(args.gpus)!=int and len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        train_sampler = DistributedSamplerResume(train_set, shuffle=args.shuffle, seed=args.seed)
        if args.max_n_vertex_per_gpu is None:
            args.batch_size = int(args.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.batch_size}')
    else:
        args.local_rank = -1
        train_sampler = None

    if args.local_rank <= 0:
        if args.max_n_vertex_per_gpu is not None:
            print_log(f'Dynamic batch enabled. Max number of vertex per GPU: {args.max_n_vertex_per_gpu}')
        if args.pretrain_ckpt:
            print_log(f'Loaded pretrained checkpoint from {args.pretrain_ckpt}')
        print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=(args.shuffle and train_sampler is None),
                              sampler=train_sampler,
                              collate_fn=train_set.collate_fn,
                              worker_init_fn=lambda x: np.random.seed(args.seed + x))
    if valid_set is not None:
        valid_loader = DataLoader(valid_set, batch_size=args.valid_batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=valid_set.collate_fn,
                                  shuffle=False)
    else:
        valid_loader = None
    trainer = create_trainer(model, train_loader, valid_loader, config, 
                             resume_state=torch.load(args.pretrain_state) if args.pretrain_state else None)
    if args.local_rank <= 0: # only log on the main process
        print_log(f"Saving model checkpoints to: {config.save_dir}")
        os.makedirs(config.save_dir, exist_ok=True)
        with open(os.path.join(config.save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        if args.use_wandb:
            wandb.init(
                entity="aidd-lilab",
                dir=config.save_dir,
                settings=wandb.Settings(start_method="fork"),
                project=f"ATOMICA-LP_PDBBind-apo&holo",
                name=args.run_name,
                config=vars(args),
            )
    trainer.train(args.gpus, args.local_rank, use_wandb=args.use_wandb, use_raytune=args.use_raytune)

    return trainer.topk_ckpt_map


if __name__ == '__main__':
    args = parse()
    main(args)
