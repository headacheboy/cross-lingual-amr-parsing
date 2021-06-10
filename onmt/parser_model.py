import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from onmt.transformer import build_embeddings, build_decoder

import onmt
from utils.misc import aeq
from utils.loss import LabelSmoothingLoss
from onmt.sublayer import PositionwiseFeedForward

class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=768, heads=8, d_ff=2048, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.sublayer.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
    
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
        self.att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        input_norm = self.att_layer_norm(inputs)
        outputs, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask)
        inputs = self.dropout(outputs) + inputs
        
        input_norm = self.ffn_layer_norm(inputs)
        outputs = self.feed_forward(input_norm)
        inputs = outputs + inputs
        return inputs

class MyTransformerEncoder(nn.Module):

    def __init__(self, num_layers=6, d_model=512, heads=8, d_ff=2048,
                dropout=0.1, embeddings=None, tknzr=None, contextual_embeddings=None):
        super(MyTransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.padding_idx = tknzr.pad_token_id
        self.embeddings = embeddings
        self.transformer = nn.ModuleList([MyTransformerEncoderLayer(d_model, heads, d_ff, dropout)
        for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.contextual_embeddings = contextual_embeddings
        if self.contextual_embeddings is not None:
            self.linear_map = nn.Linear(768+512, 512, bias=False)

    def forward(self, src, attn_mask=None):
        # src: seq_len, batch
        # NOTE: positional encoding requires the src shape: [seq_len, batch], do not transpose before embeddings
        # out: seq_len, batch, dim

        if self.contextual_embeddings is not None:
            emb = self.embeddings(src)
            #assert attn_mask is not None
            with torch.no_grad():
                contextual_emb, _ = self.contextual_embeddings(src.permute(1, 0), attn_mask)
                contextual_emb = contextual_emb.detach()
            tmp_emb = torch.cat([emb, contextual_emb.permute(1, 0, 2)], dim=-1)
            emb = self.linear_map(tmp_emb)
        
        else:
            emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src.transpose(0, 1)
        padding_idx = self.padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        return None, out.permute(1, 0, 2).contiguous(), emb


class ParserModel(nn.Module):
    def __init__(self, encoder, model_opt, fields, tknzr, lang_num, decoder=None, generator=None, contextual_embeddings=None):
        super(ParserModel, self).__init__()
        #self.encoder = encoder
        self.lang_num = lang_num
        self.encoder = MyTransformerEncoder(embeddings=encoder, tknzr=tknzr, contextual_embeddings=contextual_embeddings)
        if decoder is not None:
            self.init(model_opt)
            self.decoder = decoder
            self.generator = generator
        else:
            self.decoder, self.generator = self.build_base_decoder(model_opt, fields)
            #self.linear_map = nn.Linear(768, model_opt.dec_rnn_size)
            self.init(model_opt)
        self.decoder.src_pad_id = tknzr.pad_token_id
        #self.linear_map = nn.Linear(model_opt.enc_rnn_size, model_opt.enc_rnn_size) # for kd_type=1
        #self.gt_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=1)
        self.gt_criterion = LabelSmoothingLoss(0.1, len(fields['tgt2'].vocab), ignore_index=1)
        self.translate_bos_id = fields['tgt2'].vocab.stoi['<s>']
        self.tgt_pad_id = fields['tgt2'].vocab.stoi['<blank>']
        self.src_pad_id = fields['src'].vocab.stoi['<blank>']
        assert self.tgt_pad_id == self.src_pad_id
        assert self.tgt_pad_id == 1
        '''
        for p in self.linear_map.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
                '''

    def build_base_decoder(self, model_opt, fields):
        tgt_dict = fields['tgt2'].vocab
        tgt_embeddings = build_embeddings(model_opt, tgt_dict, for_encoder=False)
        #model_opt.dec_layers = 4
        decoder = build_decoder(model_opt, tgt_embeddings)
        generator = nn.Linear(model_opt.dec_rnn_size, len(fields["tgt2"].vocab), bias=False)

        if model_opt.share_decoder_embeddings:
            generator.weight = decoder.embeddings.word_lut.weight
        return decoder, generator

    def init(self, model_opt):
        if model_opt.param_init != 0.0:
            for p in decoder.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for n, p in self.named_parameters():
                if n.find("contextual_embeddings") != -1:
                    continue
                if p.dim() > 1:
                    xavier_uniform_(p)
        
    
    def forward(self, x, x_attn, tgt, translate=False, src=None, cal_loss=False, beta=1.0):
        # x: batch, seq_len
        # x_attn: batch, seq_len
        # tgt: batch, seq_len
        #memory_bank, pooled_output = self.encoder(x, x_attn)
        x = x.permute(1, 0)
        tgt = tgt.permute(1, 0)
        _, memory_bank, _ = self.encoder(x, x_attn)
        #memory_bank = self.linear_map(memory_bank)
        self.decoder.init_state(x, memory_bank)
        ground_truth_tgt = tgt[1:]
        tgt = tgt[:-1]
        seq_len, batch = tgt.shape
        tgt = tgt.unsqueeze(2).repeat(1, 1, self.lang_num).view(seq_len, batch*self.lang_num)
        ground_truth_tgt = ground_truth_tgt.unsqueeze(2).repeat(1, 1, self.lang_num).view(seq_len, batch*self.lang_num)
        dec_out, attns = self.decoder(tgt)
        dec_final = self.generator(dec_out)
        
        translate_dec_out = None
        if translate:
            bos = src.new_ones([1, batch]) * self.translate_bos_id
            translate_tgt = torch.cat([bos, src], dim=0)
            translate_tgt = translate_tgt[:-1]
            translate_dec_out, _ = self.decoder(translate_tgt)
            translate_dec_out = self.generator(translate_dec_out)

        if cal_loss:
            vocab_size = dec_final.shape[-1]
            gt_loss = self.gt_criterion(dec_final.view(-1, vocab_size), ground_truth_tgt.view(-1))
            gt_loss = gt_loss.view(seq_len, batch*self.lang_num)
            mask = (ground_truth_tgt == self.tgt_pad_id)
            token_num = torch.sum(1 - mask, dim=0).float()
            gt_loss = gt_loss.masked_fill(mask, 0)
            gt_loss_per_token = torch.mean(torch.sum(gt_loss, dim=0) / token_num.unsqueeze(0))
            #gt_loss_per_token = torch.mean(torch.sum(gt_loss, dim=0))
            pred = dec_final.argmax(dim=2)
            correct = (pred == ground_truth_tgt).view(seq_len, batch*self.lang_num)
            correct = correct.masked_fill(mask, 0)

            if translate:
                src_len,  src_batch = src.shape
                src_mask = (src == self.src_pad_id)
                src_loss = self.gt_criterion(translate_dec_out.view(-1, vocab_size), src.contiguous().view(-1))
                src_loss = src_loss.view(src_len, src_batch)
                src_loss = src_loss.masked_fill(src_mask, 0)
                src_token_num = torch.sum(1 - src_mask, dim=0).float()
                src_loss_per_token = torch.mean(torch.sum(src_loss, dim=0) / src_token_num.unsqueeze(0))
                #src_loss_per_token = torch.mean(torch.sum(src_loss, dim=0))
                gt_loss_per_token = gt_loss_per_token + beta * src_loss_per_token

            return gt_loss_per_token, torch.sum(correct), torch.sum(token_num)

        return dec_final, attns, memory_bank, translate_dec_out
