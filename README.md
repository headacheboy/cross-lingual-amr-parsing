# cross-lingual-amr-parsing

This is the implementation of [Making Better Use of Bilingual Information for Cross-Lingual AMR Parsing](https://arxiv.org/abs/2106.04814) in Findings of ACL 2021

### Dependency

Before training, download the pretrained S2S-AMR-Parser in [this repo](https://github.com/xdqkid/S2S-AMR-Parser)

The training data in English we used is AMR 2.0[(here)](https://catalog.ldc.upenn.edu/LDC2017T10). To get training data and development data in DE, IT, ES and ZH, use MarianMT[(here)](https://huggingface.co/transformers/model_doc/marian.html) or other machine translation systems. 

The test data in DE, IT, ES and ZH can be found [here](https://catalog.ldc.upenn.edu/LDC2020T07). 

### Preprocss and Postprocess

Use codes in `AMR_scripts/` to preprocess and postprocess the AMR datasets. 

**Preprocessing:**

```
python var_free_amrs.py -f sample_input/sample.txt
```

This will produce a text file and an AMR sequence file. Follow [this repo](https://github.com/xdqkid/S2S-AMR-Parser) to apply tokenization and BPE to original text file and AMR sequence file. Translate the text file to get *{lang}_train.txt* and *{lang}_dev.txt* (DE, IT, ES and ZH).

**Postprocessing:**

first remove BPE of outputs
```
sed -r 's/(@@ )|(@@ ?$)//g' sample.txt.bpe > sample.txt
```
then run the code
```
python postprocess_AMRs.py -f sample.txt
python reformat_single_amrs.py -f sample.txt.restore.final -e .form
```

The AMR file for evaluation is the **sample.txt.restore.final.form**

### Train and Predict

Here is a command demo for training (several path should be changed in *train.py:L234-237*)

```
python train.py --model s2s_amr_parser_path --prefix train_data_folder_path --prefix_dev dev_data_folder_path --save_prefix save_folder_path --xlm_r_path xlmr_folder_path
```

and predicting

```
python cross_translate.py --decode_extra_length 1000 --minimal_relative_prob 0.01 --gpu 0 --src your_input_file_path --translate_input your_eng_input_file_path --output your_output_path --model_path your_model_path --xlmr_path xlmr_folder_path
```

Notice that model_path for predicting is the model trained with this code instead of the S2S-AMR-Parser. Our temporary best model is in [https://drive.google.com/file/d/1SOJ0fiXpWUCkstBVq-6G9-ed5xjv330w/view?usp=sharing](https://drive.google.com/file/d/1SOJ0fiXpWUCkstBVq-6G9-ed5xjv330w/view?usp=sharing)

### Acknowledgements

We adapt the codes from [S2S-AMR-Parser](https://github.com/xdqkid/S2S-AMR-Parser) and [RikVN/AMR](https://github.com/RikVN/AMR). Thanks to their open-source projects. 
