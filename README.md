# Train Once, Deploy Anywhere: Matryoshka Representation Learning for Multimodal Recommendation

This repository is the PyTorch implementation for EMNLP 2024 Findings paper:

**Train Once, Deploy Anywhere: Matryoshka Representation Learning for Multimodal Recommendation [[Paper](https://arxiv.org/abs/2409.16627)][[Code](https://github.com/yueqirex/fMRLRec)]** (BibTex citation at the bottom, "\*" denotes equal contribution)



Yueqi Wang*, Zhenrui Yue*, Huimin Zeng, Dong Wang†, Julian McAuley. Train Once, Deploy Anywhere: Matryoshka Representation Learning for Multimodal Recommendation.

<img src=media/fig_arch.png width=1000>


## Requirements

Numpy, pandas, pytorch etc. For our detailed running environment see requirements.txt


## How to train fMRLRec
The command below specifies the training of fMRLRec on Beauty dataset.
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_code=beauty --lru_dropout=0.6 --lru_attn_dropout=0.6 --lru_r_min=0.0 --lru_r_max=0.1 --mrl_hidden_sizes="8,16,32,64,128,256,512,1024"
```

Execute the above command (with arguments) to train fMRLRec, select ```dataset_code``` from beauty, clothing, sports and toys. Set ```lru_attn_dropout``` to control LRU (Linear Recurrent Units) dropout and ```lru_dropout``` for feed-forward network dropout; ```lru_r_min``` and ```lru_r_max``` are the minimum and maximum radius of LRU ring-initialization; ```mrl_hidden_sizes``` are Matryoshka representation sizes used. Training results will be automatically saved in ```./experiments``` folder.

## How to evaluate fMRLRec
The command below specifies the training of fMRLRec on Beauty dataset.
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset_code=beauty --lru_dropout=0.6 --lru_attn_dropout=0.6 --lru_r_min=0.0 --lru_r_max=0.1
```

Execute the above command (with arguments) to evaluate different model sizes of fMRLRec. Evaluation results of different model sizes will be automatically saved to the specific experiment folder under ```./experiments``` with the name ```test_mrl_metrics.json```.

## Performance

The table below reports our main performance results, with best results marked in bold and second best results underlined. For training and evaluation details, please refer to our paper.

<img src=media/main_table.png width=1000>


## Citation
Please consider citing the following paper if you use our methods in your research:
```bib
@inproceedings{wang2024train,
  title={Train Once, Deploy Anywhere: Matryoshka Representation Learning for Multimodal Recommendation},
  author={Wang, Yueqi and Yue, Zhenrui and Zeng, Huimin and Wang, Dong and McAuley, Julian},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  year={2024}
}
```




