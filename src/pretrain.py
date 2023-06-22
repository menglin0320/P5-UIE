import collections
import os
import random
from pathlib import Path
import logging
import shutil
from packaging import version

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from src.tokenization import P5Tokenizer
from param import parse_args
from pretrain_data import get_loader
from utils import LossMeter
from dist_utils import reduce_dict
import json
import gzip
import pickle
_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

from trainer_base import TrainerBase

# The Trainer inherits TrainerBase in trainer_base.py
class Trainer(TrainerBase):
    def __init__(self, args, tokenizer=None, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        assert args.whole_word_embed
        from pretrain_model import P5Pretraining

        model_kwargs = {}
        model_class = P5Pretraining
        config = self.create_config()
        self.split = args.train
        if args.load is None:
            self.tokenizer = self.create_tokenizer()
        else:
            self.tokenizer = tokenizer
        
        self.model = self.create_model(model_class, config, **model_kwargs)
        
        if not tokenizer is None:
            self.tokenizer = tokenizer
        if self.args.mean_regularize:
            self.old_embed_norm = torch.norm(torch.mean(self.model.get_output_embeddings().weight.data[:self.tokenizer.sp_model.get_piece_size()], dim=0))
        if 'p5' in self.args.tokenizer and args.continuous_embed:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
            self.initialize_mov_len_embeds()
        self.model.tokenizer = self.tokenizer
        print('vocab size is: ', self.tokenizer.vocab_size)


        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)
            # self.start_epoch = int(args.load.split('Epoch')[-1])
            self.start_epoch = 0
        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            if args.pretrain_embed:
                self.optim, self.lr_scheduler = self.create_embedding_optimizer_and_scheduler()
            else:
                self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def _update_embed(self, embed_inds, token_ind, avg_embed, extra_tokens = None):
        token_embed = self.model.get_input_embeddings()(embed_inds.unsqueeze(0))
        token_embed = torch.mean(token_embed, dim=1)
        if not extra_tokens is None:
            extra_embed = self.model.get_input_embeddings()(extra_tokens.unsqueeze(0))
            extra_embed = torch.mean(extra_embed, dim=1)
            token_embed = (token_embed + extra_embed) / 2
        self.model.get_input_embeddings().weight.data[token_ind,:] = token_embed[0]

    def initialize_mov_len_embeds(self):
        datamaps = load_json(os.path.join('data', self.split, 'datamaps.json'))
        user_id2name = load_pickle(os.path.join('data', self.split, 'user_id2name.pkl'))
        id2user = datamaps['id2user']
        id2item = datamaps['id2item']
        meta_data = []
        meta_dict = {}
        for meta in parse(os.path.join('data', self.split, 'meta.json.gz')):
            meta_data.append(meta)
        for i, meta_item in enumerate(meta_data):
            meta_dict[meta_item['asin']] = i

        start_idx = self.tokenizer.sp_model.get_piece_size() + self.tokenizer._extra_ids
        avg_embed = torch.mean(self.model.get_output_embeddings().weight.data, dim=0)
        for i in range(0, self.tokenizer._user_extra_ids - 1): # skip 0
            user_id = self.tokenizer._user_extra_ids - i - 1
            token_ind = i + start_idx
            # user = id2user[str(user_id)]
            user = user_id2name[str(user_id)]
            assert(self.tokenizer._convert_id_to_token(token_ind) == f'<user_{user_id}>')
            
            user_embed_str = user
            # print(self.tokenizer.tokenize(user_embed_str))
            user_embed_inds = torch.from_numpy(np.asarray(self.tokenizer.encode(user_embed_str, padding=True, truncation=True, max_length=100)))
            self._update_embed(user_embed_inds, token_ind, avg_embed)

        start_idx = start_idx + self.tokenizer._user_extra_ids
        for i in range(0, self.tokenizer._item_extra_ids - 1): # skip 0
            item_id = self.tokenizer._item_extra_ids - i - 1
            token_ind = i + start_idx
            item = id2item[str(item_id)]
            # item = str(item_id)
            item = meta_data[meta_dict[item]].get('title', item)
            # title_string = 'item title'
            assert(self.tokenizer._convert_id_to_token(token_ind) == f'<item_{item_id}>')
            item_embed_str = item

            # print(self.tokenizer.tokenize(item_embed_str))
            item_embed_inds = torch.from_numpy(np.asarray(self.tokenizer.encode(item_embed_str, padding=True, truncation=True, max_length=100)))
            # title_string_inds = torch.from_numpy(np.asarray(self.tokenizer.encode(title_string, padding=True, truncation=True, max_length=100)))
            self._update_embed(item_embed_inds, token_ind, avg_embed)

    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME

        if self.args.dry:
            results = self.evaluate_epoch(epoch=0)

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_eval_loss = 100000.

            if 't5' in self.args.backbone:
                project_name = "P5_Pretrain"

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        for epoch in range(self.args.epoch):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            # Train
            self.model.train()

            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=275)

            epoch_results = {}
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):

                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.args.clip_grad_norm)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                # print(grad_norm)
                
                
                embed_param = None
                for n, p in self.model.named_parameters():
                    if 'shared' in n:
                        embed_param = p
                assert(not embed_param is None)

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr
                if self.args.pretrain_embed:
                    if self.args.mean_regularize:
                        decay_mask = torch.ones(embed_param.size(), dtype=embed_param.dtype).to(embed_param.device)
                        cur_norm = torch.norm(torch.mean(embed_param[self.tokenizer.sp_model.get_piece_size()  + self.tokenizer._extra_ids - 4:], dim=1))
                        decay_mask[self.tokenizer.sp_model.get_piece_size()  + self.tokenizer._extra_ids - 4:, :] = self.old_embed_norm / cur_norm
                        prev_grad = torch.is_grad_enabled()
                        torch.set_grad_enabled(False)
                        embed_param.mul_(decay_mask)
                        torch.set_grad_enabled(prev_grad)
                    else:
                        decay_mask = torch.ones(embed_param.size(), dtype=embed_param.dtype).to(embed_param.device)
                        decay_mask[self.tokenizer.sp_model.get_piece_size()  + self.tokenizer._extra_ids - 4:, :] = (1- lr * self.args.weight_decay)
                        prev_grad = torch.is_grad_enabled()
                        torch.set_grad_enabled(False)
                        embed_param.mul_(decay_mask)
                        torch.set_grad_enabled(prev_grad)
                    
                # elif epoch == 0:
                #     decay_mask = torch.ones(embed_param.size(), dtype=embed_param.dtype).to(embed_param.device)
                #     self.model.module.shared.weight.grad[self.tokenizer.sp_model.get_piece_size() + self.tokenizer._extra_ids + 1:, :] *= (step_i/len(self.train_loader))
                    # print(self.model.module.shared.weight.grad[40])
                    # print(self.model.module.shared.weight[40])

                if self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()
                
                    
                    
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                # self.model.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 200:
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()

            dist.barrier()

            results = reduce_dict(epoch_results, average=False)
            if self.verbose:
                train_loss = results['total_loss']
                train_loss_count = results['total_loss_count']

                avg_train_loss = train_loss / train_loss_count
                losses_str = f"Train Loss: {avg_train_loss:.3f}\n"

                for name, loss in results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss/loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                losses_str += '\n'
                print(losses_str)

            dist.barrier()

            if epoch > 10:
                # Validation
                valid_results = self.evaluate_epoch(epoch=epoch)

                valid_results = reduce_dict(valid_results, average=False)
                if self.verbose and step_i % 200:
                    valid_loss = valid_results['total_loss']
                    valid_loss_count = valid_results['total_loss_count']

                    avg_valid_loss = valid_loss / valid_loss_count
                    losses_str = f"Valid Loss: {avg_valid_loss:.3f}\n"

                    for name, loss in valid_results.items():
                        if name[-4:] == 'loss':
                            loss_count = int(valid_results[name+'_count'])
                            if loss_count > 0:
                                avg_loss = loss / loss_count
                                losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                    losses_str += '\n'
                    print(losses_str)

                dist.barrier()

                if self.verbose:
                    # Save
                    if avg_valid_loss < best_eval_loss:
                        best_eval_loss = avg_valid_loss
                        self.save("BEST_EVAL_LOSS")
                    self.save("Epoch%02d" % (epoch + 1))

                dist.barrier()
            else:
                # Skip validation
                print("Skip validation for Epoch%02d" % (epoch + 1))
                self.save("Epoch%02d" % (epoch + 1))
                
                dist.barrier()

    def evaluate_epoch(self, epoch):
        LOSSES_NAME = self.args.LOSSES_NAME

        epoch_results = {}
        for loss_name in LOSSES_NAME:
            epoch_results[loss_name] = 0.
            epoch_results[f'{loss_name}_count'] = 0

        self.model.eval()
        with torch.no_grad():
            if self.verbose:
                loss_meter = LossMeter()
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

                pbar = tqdm(total=len(self.val_loader), ncols=275)

            for step_i, batch in enumerate(self.val_loader):

                if self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.valid_step(batch)

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 200:
                    desc_str = f'Valid Epoch {epoch} |'
                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)
                dist.barrier()

            if self.verbose:
                pbar.close()
            dist.barrier()

            return epoch_results


def main_worker(gpu, args):
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.continuous_embed:
        datamaps = load_json(os.path.join('data', args.train, 'datamaps.json'))
        user_extra_ids = len(datamaps['user2id']) + 1 # index start from 1
        item_extra_ids = len(datamaps['item2id']) + 1 
    else:
        user_extra_ids = 0
        item_extra_ids = 0
    tokenizer = P5Tokenizer.from_pretrained(
            args.backbone, 
            max_length=args.max_text_length, 
            do_lower_case=args.do_lower_case,
            user_extra_ids=user_extra_ids,
            item_extra_ids=item_extra_ids
            )
    
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')
    # define the prompts used in training
    if args.train == 'yelp':
        train_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
        'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
        'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9'],
        'review': ['4-1', '4-2'],
        'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
        }
    else:
        if args.pretrain_embed:
            train_task_list = {'relate_inout_embeds': ['6-1'],
                               'learn_title': ['7-1'],
                               'learn_category': ['8-1'],
                               'learn_brand': ['9-1'],
                               'learn_price': ['10-1'],
                               'learn_user': ['11-1']}
        else:
            train_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
            'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
            'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11'],
            'review': ['4-1', '4-2', '4-3'],
            'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7'],
            'relate_inout_embeds': ['6-1'],
            'learn_title': ['7-1'],
            'learn_category': ['8-1'],
            'learn_brand': ['9-1'],
            'learn_price': ['10-1'],
            'learn_user': ['11-1']
            }
    # define sampling numbers for each group of personalized prompts (see pretrain_data.py)
    # if greater than 1, a data sample will be used for multiple times with different prompts in certain task family
    if args.pretrain_embed:
        train_sample_numbers = {'relate_inout_embeds': 1, 'learn_title': 2, 'learn_category': 4,
                                'learn_brand': 2, 'learn_price': 2, 'learn_user': 2}
    else:
        train_sample_numbers = {'rating': 1, 'sequential': (5, 5, 10), 'explanation': 1, 'review': 1, 'traditional': (10, 5),
                                'relate_inout_embeds': 1, 'learn_title': 1, 'learn_category': 2,
                                'learn_brand': 1, 'learn_price': 1, 'learn_user': 1}

    train_loader = get_loader(
        args,
        train_task_list,
        train_sample_numbers,
        tokenizer = tokenizer,
        split=args.train, 
        mode='train',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
    )

    print(f'Building val loader at GPU {gpu}')
    # define the prompts used in validation
    if args.valid == 'yelp':
        val_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
        'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
        'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9'],
        'review': ['4-1', '4-2'],
        'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7'],

        }
    else:
        if args.pretrain_embed:
            val_task_list = {'relate_inout_embeds': ['6-1'],
                             'learn_title': ['7-1'],
                             'learn_category': ['8-1'],
                             'learn_brand': ['9-1'],
                             'learn_price': ['10-1'],
                             'learn_user': ['11-1']}
        else:
            val_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
            'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
            'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11'],
            'review': ['4-1', '4-2', '4-3'],
            'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
            }
    if args.pretrain_embed:
        val_sample_numbers = {'relate_inout_embeds': 1, 'learn_title': 1, 'learn_category': 4,
                                'learn_brand': 2, 'learn_price': 2, 'learn_user': 2}
    else:
        val_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
    val_loader = get_loader(
        args,
        val_task_list,
        val_sample_numbers,
        tokenizer = tokenizer,
        split=args.valid, 
        mode='val',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
    )

    trainer = Trainer(args, tokenizer, train_loader, val_loader, train=True)
    trainer.train()


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    if args.local_rank in [0, -1]:
        print(args)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
    if args.local_rank in [0, -1]:
        print(LOSSES_NAME)
    LOSSES_NAME.append('total_loss')

    args.LOSSES_NAME = LOSSES_NAME

    comments = []
    dsets = []
    if 'toys' in args.train:
        dsets.append('toys')
    if 'beauty' in args.train:
        dsets.append('beauty')
    if 'sports' in args.train:
        dsets.append('sports')
    if 'yelp' in args.train:
        dsets.append('yelp')
    comments.append(''.join(dsets))
    if args.backbone:
        comments.append(args.backbone)
    comments.append(''.join(args.losses.split(',')))
    if args.comment != '':
        comments.append(args.comment)
    comment = '_'.join(comments)

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M')

    project_dir = Path(__file__).resolve().parent.parent

    if args.local_rank in [0, -1]:
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'
        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
