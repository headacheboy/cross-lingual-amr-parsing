import numpy as np
import torch
import random
import math

def tokenize(vocab, snt):
    for i in range(len(snt)):
        snt[i] = vocab.stoi.get(snt[i], vocab.stoi['<unk>'])
    return snt

class SrcSent():
    def __init__(self, snt, tknzr, src_vocab):
        self.str_snt = " ".join(snt.split())
        self.bpe_snt = snt.split() + ["</s>"]
        if src_vocab is not None:
            self.bpe_snt = tokenize(src_vocab, self.bpe_snt)
        self.snt = tknzr(snt.replace("@@ ", ""))   # for XLMRobertaTokenizer()
        self.attn_mask = self.snt['attention_mask']
        self.snt = self.snt['input_ids']
        self.snt_len = len(self.snt)
    
    def __len__(self):
        return len(self.snt)

class InferenceDataLoader():
    def __init__(self, src_filename, batch_size, tknzr, src_vocab):
        self.batch_size = batch_size
        self.tknzr_pad_id = tknzr.pad_token_id
        self.src_ls = []
        if type(src_filename) == list:
            self.load_data_list(src_filename, tknzr, src_vocab)
        else:
            self.load_data(src_filename, tknzr, src_vocab)
    
    def load_data(self, src_filename, tknzr, src_vocab):
        f = open(src_filename, encoding='utf-8')
        idx = 0
        for line in f:
            idx += 1
            print(idx)
            self.src_ls.append(SrcSent(line, tknzr, src_vocab))
    
    def load_data_list(self, src_filename_ls, tknzr, src_vocab):
        eng_src_ls = []
        tgt_src_ls = []
        f = open(src_filename_ls[0], encoding='utf-8')
        idx = 0
        for line in f:
            idx += 1
            print(idx)
            eng_src_ls.append(SrcSent(line, tknzr, src_vocab))
        f = open(src_filename_ls[1], encoding='utf-8')
        idx = 0
        for line in f:
            idx += 1
            print(idx)
            tgt_src_ls.append(SrcSent(line, tknzr, src_vocab))
            tgt_src_ls[-1].snt.extend(eng_src_ls[idx-1].snt[1:])
            tgt_src_ls[-1].attn_mask.extend(eng_src_ls[idx-1].attn_mask[1:])
        assert len(tgt_src_ls) == len(eng_src_ls)
        self.src_ls = tgt_src_ls

    def batch(self):
        data = []
        idx_span = 0
        tmp_max_src = 0

        for i in range(len(self.src_ls)):
            src_num = len(self.src_ls[i])
            if src_num > tmp_max_src:
                tmp_max_src = src_num
            if src_num * idx_span > self.batch_size:
                if len(data) != 0:
                    yield self.batchify(data)
                data = [self.src_ls[i]]
                idx_span = 1
                tmp_max_src = src_num
            else:
                idx_span += 1
                data.append(self.src_ls[i])
        if len(data) != 0:
            yield self.batchify(data)
    
    def batchify(self, data):
        max_src_length = 0
        for datum in data:
            max_src_length = max(max_src_length, len(datum.snt))
        max_src_length = min(max_src_length, 512)

        xlm_ls = []
        attn_ls = []
        str_snt = []
        for datum in data:
            xlm_ls.append(datum.snt[:max_src_length] + [self.tknzr_pad_id] * (max_src_length - len(datum.snt))) # for tgt_language-AMR
            attn_ls.append(datum.attn_mask[:max_src_length] + [0] * (max_src_length - len(datum.attn_mask)))
            str_snt.append(datum.str_snt)
        return torch.Tensor(xlm_ls).long(), torch.Tensor(attn_ls).long(), str_snt

class DataLoader():
    def __init__(self, src_filename_ls, tgt_filename, batch_size, tknzr, src_vocab, tgt_vocab, train, world_size=1, 
                    rank=0, flatten=1, mask_rate=0.0):
        # src_ls[i] = samples in the i-th language
        self.mask_rate = mask_rate
        self.max_src_num = 150
        self.rank = rank
        self.language_num = len(src_filename_ls)
        self.world_size = world_size
        self.batch_size = batch_size
        self.src_pad_id = src_vocab.stoi["<blank>"]
        self.tknzr_pad_id = tknzr.pad_token_id
        self.tknzr_mask_id = tknzr.mask_token_id
        self.tgt_pad_id = tgt_vocab.stoi["<blank>"]
        self.train = train
        self.src_ls = [[] for _ in range(len(src_filename_ls))] 
        self.tgt_ls = []
        for i in range(len(src_filename_ls)):
            f = open(src_filename_ls[i], encoding='utf-8')
            for line in f:
                self.src_ls[i].append(SrcSent(line, tknzr, src_vocab=src_vocab if i == 0 else None))
        for i in range(1, len(src_filename_ls)):
            for j in range(len(self.src_ls[i])):
                self.src_ls[i][j].bpe_snt = self.src_ls[0][j].bpe_snt
        f = open(tgt_filename, encoding='utf-8')
        for line in f:
            self.tgt_ls.append(["<s2>"] + line.split() + ["</s>"])
            self.tgt_ls[-1] = tokenize(tgt_vocab, self.tgt_ls[-1])
            
        if flatten == 1:
            # flatten the src_ls 
            new_src_ls = [[]]
            new_tgt_ls = []
            for i in range(len(self.tgt_ls)):
                for j in range(self.language_num):
                    new_src_ls[0].append(self.src_ls[j][i])
                    new_tgt_ls.append(self.tgt_ls[i])
            self.src_ls = new_src_ls
            self.tgt_ls = new_tgt_ls
            self.language_num = 1
        elif flatten == 2:
            # flatten the src_ls, concat the Eng snt and the foreign language snt
            self.max_src_num = 300
            new_src_ls = [[]]
            new_tgt_ls = []
            for i in range(len(self.tgt_ls)):
                for j in range(1, self.language_num):
                    self.src_ls[j][i].snt.extend(self.src_ls[0][i].snt[1:]) # add Eng snt, remove BOS of eng snt
                    self.src_ls[j][i].attn_mask.extend(self.src_ls[0][i].attn_mask[1:])
                    new_src_ls[0].append(self.src_ls[j][i])
                    new_tgt_ls.append(self.tgt_ls[i])
            self.src_ls = new_src_ls
            self.tgt_ls = new_tgt_ls
            self.language_num = 1
        print("data finished")
    
    def batch(self):
        idx_ls = list(range(len(self.tgt_ls)))
        if self.train:
            random.shuffle(idx_ls)
        for i in range(len(idx_ls)):
            idx_ls[i] = [i, max(len(self.tgt_ls[i]), len(self.src_ls[0][i]))]
        idx_ls.sort(key=lambda x: x[1], reverse=True) # idx_ls[i]: [idx, len]

        data = []
        idx_span = 0
        tmp_max_src = 0
        tmp_max_tgt = 0

        yield_data = []

        if self.world_size > 1:
            length = len(idx_ls)
            for ptr in range(length):
                idx = idx_ls[ptr][0]
                src_num = max([len(self.src_ls[t][idx]) for t in range(self.language_num)])
                tgt_num = len(self.tgt_ls[idx])
                if self.train and src_num > self.max_src_num:
                    continue
                tmp_max_src = max(tmp_max_src, src_num)
                tmp_max_tgt = max(tmp_max_tgt, tgt_num)
                if len(data) < self.world_size:
                    data.append([self.src_ls[t][idx] for t in range(self.language_num)] + [self.tgt_ls[idx]])
                    continue
                flag = 0 if (len(data) + 1) % self.world_size == 0 else 1
                if tmp_max_src * ((len(data) + 1) // self.world_size + flag) * self.language_num + \
                        tmp_max_tgt * self.language_num * \
                            ((len(data) + 1) // self.world_size + flag) > self.batch_size:
                    if len(data) >= self.world_size:
                        yield_data.append(data)
                    data = [[self.src_ls[t][idx] for t in range(self.language_num)] + [self.tgt_ls[idx]]]
                    tmp_max_src = src_num
                    tmp_max_tgt = tgt_num
                else:
                    data.append([self.src_ls[t][idx] for t in range(self.language_num)] + [self.tgt_ls[idx]])
            if len(data) >= self.world_size:
                yield_data.append(data)
            if self.train:
                random.shuffle(yield_data)
            for ele in yield_data:
                yield self.batchify(ele)
        else:
            for i in range(len(idx_ls)):
                idx = idx_ls[i][0]
                src_num = max([len(self.src_ls[t][idx]) for t in range(self.language_num)])
                tgt_num = len(self.tgt_ls[idx])
                if self.train and src_num > self.max_src_num:
                    continue
                if src_num > tmp_max_src:
                    tmp_max_src = src_num
                if tgt_num > tmp_max_tgt:
                    tmp_max_tgt = tgt_num
                if (tmp_max_src + tmp_max_tgt) * (idx_span+1) * self.language_num > self.batch_size:
                    if len(data) != 0:
                        yield_data.append(data)
                    data = [[self.src_ls[t][idx] for t in range(self.language_num)] + [self.tgt_ls[idx]]]
                    idx_span = 1
                    tmp_max_src = src_num
                    tmp_max_tgt = tgt_num
                else:
                    idx_span += 1
                    data.append([self.src_ls[t][idx] for t in range(self.language_num)] + [self.tgt_ls[idx]])
            if len(data) != 0:
                yield_data.append(data)
            if self.train:
                random.shuffle(yield_data)
            for ele in yield_data:
                yield self.batchify(ele)
            
    def batchify(self, data):
        # data: list
        # data[i]: [SrcSent_1, ..., SrcSent_5, tgt]
        # tgt is a list
        idx_ls = list(range(len(data)))
        length = len(data)
        per_length = [length // self.world_size for _ in range(self.world_size)]
        if length % self.world_size != 0:
            for i in range(length % self.world_size):
                per_length[i] += 1
        for i in range(1, self.world_size):
            per_length[i] += per_length[i-1]
        batch_size = len(data)
        max_src_length_1 = max_src_length_2 = max_tgt_length = 0
        for idx, datum in enumerate(data):
            if self.rank == 0:
                if idx >= per_length[0]:
                    continue
            else:
                if idx < per_length[self.rank-1] or idx >= per_length[self.rank]:
                    continue
            for i in range(self.language_num):
                max_src_length_1 = max(max_src_length_1, len(datum[i].snt))
            max_src_length_2 = max(max_src_length_2, len(datum[0].bpe_snt))
            max_tgt_length = max(max_tgt_length, len(datum[-1]))

        xlm_ls = []
        src_ls = []
        tgt_ls = []
        attn_ls = []
        if self.world_size == 1:
            for datum in data:
                for i in range(self.language_num):
                    new_snt = datum[i].snt
                    new_attn_mask = datum[i].attn_mask
                    for t in range(len(new_snt)-1, datum[i].snt_len-1, -1):
                        if random.random() < self.mask_rate:
                            #new_snt[t] = self.tknzr_mask_id    # for masking
                            new_snt = new_snt[:t] + new_snt[t+1:]   # for deleting
                            new_attn_mask = new_attn_mask[:t] + new_attn_mask[t+1:]
                    xlm_ls.append(new_snt + [self.tknzr_pad_id] * (max_src_length_1 - len(new_snt)))
                    attn_ls.append(new_attn_mask + [0] * (max_src_length_1 - len(new_attn_mask)))
                src_ls.append(datum[0].bpe_snt + [self.src_pad_id] * (max_src_length_2 - len(datum[0].bpe_snt)))
                tgt_ls.append(datum[-1] + [self.tgt_pad_id] * (max_tgt_length - len(datum[-1])))
            return torch.Tensor(xlm_ls).long(), torch.Tensor(attn_ls).long(), torch.Tensor(src_ls).long(), torch.Tensor(tgt_ls).long()
        else:
            for idx, datum in enumerate(data):
                if self.rank == 0:
                    if idx >= per_length[0]:
                        continue
                else:
                    if idx < per_length[self.rank-1] or idx >= per_length[self.rank]:
                        continue
                for i in range(self.language_num):
                    new_snt = datum[i].snt
                    new_attn_mask = datum[i].attn_mask
                    for t in range(len(new_snt)-1, datum[i].snt_len-1, -1):
                        if random.random() < self.mask_rate:
                            #new_snt[t] = self.tknzr_mask_id    # for masking
                            new_snt = new_snt[:t] + new_snt[t+1:]   # for deleting
                            new_attn_mask = new_attn_mask[:t] + new_attn_mask[t+1:]
                    xlm_ls.append(new_snt + [self.tknzr_pad_id] * (max_src_length_1 - len(new_snt)))
                    attn_ls.append(new_attn_mask + [0] * (max_src_length_1 - len(new_attn_mask)))
                src_ls.append(datum[0].bpe_snt + [self.src_pad_id] * (max_src_length_2 - len(datum[0].bpe_snt)))
                tgt_ls.append(datum[-1] + [self.tgt_pad_id] * (max_tgt_length - len(datum[-1])))
            return torch.Tensor(xlm_ls).long(), torch.Tensor(attn_ls).long(), torch.Tensor(src_ls).long(), torch.Tensor(tgt_ls).long()
