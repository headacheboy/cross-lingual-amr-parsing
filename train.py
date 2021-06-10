from collections import defaultdict
import numpy as np
import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig
from inputters.data_loader import DataLoader
import configargparse

import onmt.opts as opts
import onmt.transformer as nmt_model
import onmt.parser_model as parser_model

import torch.distributed as dist
import torch.multiprocessing as mp

import time
import os
import random
import datetime

def update_lr(step, d_model, base_lr, warmup, optim):
    lr = base_lr * d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    for idx, param_group in enumerate(optim.param_groups):
        if idx == 0:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * 0.5
    return lr

def main(rank, opt, dummy_opt):

    print("process begin, rank:", rank)

    torch.manual_seed(opt.torch_seed)
    torch.cuda.manual_seed_all(opt.torch_seed)
    random.seed(opt.seed)
    torch.cuda.set_device(rank)

    max_epoch = opt.max_epoch # 250
    max_step = opt.max_step   # 120000
    warmup = opt.warmup
    base_lr = opt.base_lr
    grad_accu = opt.grad_accu
    batch_size = opt.batch_size
    d_model = 512
    flatten = 2

    translate = (opt.use_translate == 1)
    use_xlm = (opt.use_xlm == 1)
    init = (opt.use_dec != 1)

    # for OOM error
    if translate:
        batch_size = 2500
        grad_accu = 3
    
    if use_xlm:
        batch_size = 2048
        grad_accu = 2

    prefix = opt.prefix
    prefix_dev = opt.prefix_dev
    train_ls = ["train.txt.split.bpe", "de_train.txt", "es_train.txt", "it_train.txt", "zh_train.txt"]
    src_filename_train_ls = [prefix + ele for ele in train_ls]
    tgt_filename = prefix + "train.txt.ground.truth.bpe"
    dev_ls = ['dev.txt.split.bpe', 'de_dev.txt', 'es_dev.txt', 'it_dev.txt', 'zh_dev.txt']
    src_filename_dev_ls = [prefix_dev + ele for ele in dev_ls]
    tgt_filename_dev = prefix_dev + "dev.txt.ground.truth.bpe"

    lang_num = len(train_ls) if flatten == 0 else 1

    save_prefix = opt.save_prefix

    tknzr = XLMRobertaTokenizer.from_pretrained(opt.xlm_r_path)
    if use_xlm:
        xlm_model = XLMRobertaModel.from_pretrained(opt.xlm_r_path)
    else:
        xlm_model = None

    fields, teacher_model, model_opt = nmt_model.load_test_model(opt, dummy_opt.__dict__, use_softmax=False, get_model_opt=True, init=init)

    train_data_loader = DataLoader(src_filename_train_ls, tgt_filename, batch_size=batch_size, tknzr=tknzr, src_vocab=fields['src'].vocab, tgt_vocab=fields['tgt2'].vocab, 
        train=True, world_size=opt.world_size, rank=rank, flatten=flatten, mask_rate=0.0)
    dev_data_loader = DataLoader(src_filename_dev_ls, tgt_filename_dev, batch_size=batch_size, tknzr=tknzr, src_vocab=fields['src'].vocab, tgt_vocab=fields['tgt2'].vocab, 
        train=False, world_size=1, rank=0, flatten=flatten)

    device = torch.device("cuda")

    embeddings = nmt_model.build_embeddings_2(model_opt.src_word_vec_size, tknzr.pad_token_id, tknzr.vocab_size, model_opt)

    student_model = parser_model.ParserModel(embeddings, model_opt, fields, tknzr, lang_num, decoder=teacher_model.decoder, 
        generator=teacher_model.generator, contextual_embeddings=xlm_model)
    joint_model = student_model
    joint_model.to(device)
    if opt.world_size > 1:
        joint_model = torch.nn.parallel.DistributedDataParallel(joint_model, find_unused_parameters=True, device_ids=[rank], output_device=[rank])
    joint_model.train()

    other_params = []
    enc_params = []
    for k, v in joint_model.named_parameters():
        if k.find('encoder') == -1:
            other_params.append(v)
        else:
            if k.find('contextual_embeddings') == -1:
                enc_params.append(v)
    params = [
        {'params': enc_params}, 
        {'params': other_params}
    ]

    optimizer = torch.optim.Adam(params, lr=0.0001)
    optimizer.zero_grad()

    max_acc = 0.
    step = 0
    backward_step = 0
    lr = 0

    for epoch in range(max_epoch):
        start_time = time.time()
        avg_loss = 0.
        batch_num = 0
        samples_num = 0
        for data_iter in train_data_loader.batch():
            step += 1
            xlm_test, xlm_attn_mask, src_test, tgt_test = data_iter
            xlm_test = xlm_test.to(device)
            xlm_attn_mask = xlm_attn_mask.to(device)
            src_test = src_test.to(device)
            tgt_test = tgt_test.to(device)
            
            samples_num += xlm_test.shape[0]
            batch_num += xlm_test.shape[0]
            
            loss, _, _ = joint_model(x=xlm_test, x_attn=xlm_attn_mask, 
                            tgt=tgt_test, cal_loss=True, src=src_test.permute(1, 0), translate=translate, beta=0.5)

            avg_loss += loss.item() * xlm_test.shape[0]
            loss.backward()
            del loss
            if step % grad_accu == 0:
                backward_step += 1
                lr = update_lr(backward_step, d_model, base_lr, warmup, optimizer)
                optimizer.step()
                optimizer.zero_grad()
                
                if backward_step % 600 == 0:
                    end_time = time.time()
                    print("600 step, time:", end_time - start_time)
                    start_time = end_time
        end_time = time.time()
        print("Epoch {0}, step {4}, train loss:{1}, lr:{2}, time:{3}".format(epoch, avg_loss / batch_num, lr, end_time - start_time, backward_step))
        acc = validation(joint_model, dev_data_loader, device, epoch)
        if max_acc < acc:
            max_acc = acc
            if rank == 0:
                save(joint_model, save_prefix + "{0}.pth".format(epoch), model_opt, fields, opt.world_size)
                print("save")
        elif epoch % 4 == 0:
            if rank == 0:
                save(joint_model, save_prefix + "{0}.pth".format(epoch), model_opt, fields, opt.world_size)
                print("save every 4 epoch")
        if backward_step > max_step:
            break

def validation(model, dev_data_loader, device, epoch):
    model.eval()
    loss = 0
    correct = 0
    num = 0
    token_num = 0
    start = time.time()
    for data_iter in dev_data_loader.batch():
        xlm_x, xlm_attn_mask, src, tgt = data_iter
        xlm_x = xlm_x.to(device)
        xlm_attn_mask = xlm_attn_mask.to(device)
        tgt = tgt.to(device)

        num += tgt.shape[0]
        with torch.no_grad():
            tmp_loss, tmp_correct, tmp_token_num = model(xlm_x, xlm_attn_mask, tgt, cal_loss=True)

            loss += torch.mean(tmp_loss).item()
            correct += torch.sum(tmp_correct).item()
            token_num += torch.sum(tmp_token_num).item()
            del tmp_loss, tmp_correct, tmp_token_num
    print("Epoch {0}, dev loss: {1}, dev acc: {2}:, time: {3}".format(epoch, loss / num, correct / token_num, time.time() - start))
    model.train()
    return correct / token_num

def load(model, load_path):
    state = torch.load(load_path)
    model.student_model.load_state_dict(state['student_model'])

def save(model, save_path, model_opt, fields, world_size):
    if world_size > 1:
        t1 = model.module.state_dict()
    else:
        t1 = model.state_dict()
    state = {
        'student_model': t1, 
        'model_opt': model_opt, 
        'fields': fields
    }
    torch.save(state, save_path)

def init_processes(local_rank, opt, dummy_opt, backend='nccl'):
    os.environ['MASTER_ADDR'] = opt.MASTER_ADDR
    os.environ['MASTER_PORT'] = opt.MASTER_PORT
    dist.init_process_group(backend, rank=local_rank, world_size=opt.world_size)
    main(local_rank, opt, dummy_opt)

def training_opts(parser):
    group = parser.add_argument_group('Model')
    group.add('--model', '-model', dest='models', metavar='MODEL',
              nargs='+', type=str, default=[], required=True,
              help='Path to model .pt file(s). '
              'Multiple models can be specified, '
              'for ensemble decoding.')
    parser.add_argument('--MASTER_ADDR', type=str, default='localhost')
    parser.add_argument('--MASTER_PORT', type=str, default='23457')
    parser.add_argument('--world_size', type=int, default=4)

    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--max_step', type=int, default=300000)
    parser.add_argument('--warmup', type=int, default=4000)
    parser.add_argument('--base_lr', type=float, defalut=1.0)
    parser.add_argument('--grad_accu', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4096) # token batch size
    parser.add_argument('--use_xlm', type=int, default=1)
    parser.add_argument('--use_dec', type=int, default=1)
    parser.add_argument('--use_translate', type=int, default=1)

    parser.add_argument('--prefix', type=str, default="/home/caiyitao/s2s_amr_parsing/translate_data/")
    parser.add_argument('--prefix_dev', type=str, default="/home/caiyitao/s2s_amr_parsing/translate_data/")
    parser.add_argument('--save_prefix', type=str, defalut="/home/caiyitao/s2s_amr_parsing/s2s_amr_parser/model_multi/")
    parser.add_argument('--xlm_r_path', type=str, default="/home/caiyitao/s2s_amr_parsing/xlm_test/xlm_roberta/")

if __name__ == "__main__":
    
    parser = configargparse.ArgumentParser(description='translate.py', config_file_parser_class=configargparse.YAMLConfigFileParser, formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    opts.config_opts(parser)
    training_opts(parser)

    opt = parser.parse_args()
    dummy_parser = configargparse.ArgumentParser(description='translate.py')

    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.torch_seed = int(random.random() * 100000)
    opt.seed = int(random.random() * 100000)

    if opt.world_size == 1:
        main(0, opt, dummy_opt)
    else:
        print("world size: ", opt.world_size)
        mp.spawn(init_processes, args=(opt, dummy_opt), nprocs=opt.world_size)
        
