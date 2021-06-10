# cross-lingual-amr-parsing

This is the implementation of [Making Better Use of Bilingual Information for Cross-Lingual AMR Parsing](https://arxiv.org/abs/2106.04814) in Findings of ACL 2021

### Dependency

Before training, download the pretrained S2S-AMR-Parser in [this repo](https://github.com/xdqkid/S2S-AMR-Parser)

The training data in English we used is AMR 2.0[(here)](https://catalog.ldc.upenn.edu/LDC2017T10). To get training data in DE, IT, ES and ZH, use MarianMT[(here)](https://huggingface.co/transformers/model_doc/marian.html) or other machine translation system. 

The test data in DE, IT, ES and ZH can be found [here](https://catalog.ldc.upenn.edu/LDC2020T07). 

### Train and Predict

Here is a command demo for training

```
python train.py --model s2s_amr_parser_path
```

and predicting

```
python cross_translate.py --decode_extra_length 1000 --minimal_relative_prob 0.01 --gpu 0 --src your_input_file_path --output your_output_path
```

the input file should be ended with {0}\_test.txt, where {0} denotes the langauge (de, it, es, zh) 
