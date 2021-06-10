# cross-lingual-amr-parsing

This is the implementation of [Making Better Use of Bilingual Information for Cross-Lingual AMR Parsing]() in Findings of ACL 2021

### Dependency

Before training, download the pretrained S2S-AMR-Parser in [this repo](https://github.com/xdqkid/S2S-AMR-Parser)

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
