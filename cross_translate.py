#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from onmt import parser_model
import configargparse
import codecs

from utils.logging import init_logger
from inputters.dataset import make_text_iterator_from_file
import onmt.opts as opts
from onmt.cross_translator import build_translator
import onmt.transformer as nmt_model

from onmt.parser_model import ParserModel
from inputters.data_loader import InferenceDataLoader

from transformers import XLMRobertaModel, XLMRobertaTokenizer
import torch
import onmt.transformer as nmt_model

def main(opt):

  batch_size = opt.batch_size

  state = torch.load(opt.model_path)

  tknzr = XLMRobertaTokenizer.from_pretrained(opt.xlmr_path)
  xlm_model = XLMRobertaModel.from_pretrained(opt.xlmr_path)
  
  model_opt = state["model_opt"]
  
  embeddings = nmt_model.build_embeddings_2(model_opt.src_word_vec_size, tknzr.pad_token_id, tknzr.vocab_size, model_opt)

  
  model = ParserModel(embeddings, model_opt, state['fields'], tknzr, 1, contextual_embeddings=xlm_model)
  model.load_state_dict(state['student_model'])

  translator = build_translator(model, state['fields'], opt)
  out_file = codecs.open(opt.output, 'w+', 'utf-8')
  
  src_iter = InferenceDataLoader([opt.translate_template.format(opt.language), opt.src], 
                      batch_size, tknzr, state['fields']['src'].vocab).batch()
  if opt.tgt is not None:
    tgt_iter = make_text_iterator_from_file(opt.tgt)
  else:
    tgt_iter = None
  translator.translate(src_data_iter=src_iter,
                       tgt_data_iter=tgt_iter,
                       batch_size=opt.batch_size,
                       out_file=out_file)
  out_file.close()


def eval_opts(parser):
  parser.add_argument('--translate_template', type=str, default="/home/caiyt/s2s_amr_parsing/translate_{}.txt.split.bpe")
  parser.add_argument('--model_path', type=str, default="/home/caiyt/s2s_amr_parsing/s2s_amr_parser/model_multi/full_model_xlmr_dec.ckpt")
  parser.add_argument('--xlmr_path', type=str, default="/home/caiyt/s2s_amr_parsing/xlm_test/xlm_roberta/")


if __name__ == "__main__":
  parser = configargparse.ArgumentParser(
    description='translate.py',
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
  opts.config_opts(parser)
  opts.translate_opts(parser)
  eval_opts(parser)

  opt = parser.parse_args()
  opt.language = opt.src[-11:-9]
  logger = init_logger(opt.log_file)
  logger.info("Input args: %r", opt)
  main(opt)
