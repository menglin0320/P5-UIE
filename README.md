# P5-UIE

This project is mainly based on jeykigung's work
access their github repo through [P5 repo](https://github.com/jeykigung/P5.git)
> Paper link: https://arxiv.org/pdf/2203.13366.pdf


## Introduction
This work builds upon P5, a multi-task large language model-based recommender system. In addition, I incorporated user and item embeddings into the framework. Furthermore, I introduced several "attribute learning" tasks to train embeddings that are aware of attributes, thereby enhancing overall performance.

## Requirements:
- Python 3.9.7
- PyTorch 1.10.1
- transformers 4.2.1
- tqdm
- numpy
- sentencepiece
- pyyaml


## Usage

### Run the experiment
0. Clone this repo

    ```
    git clone https://github.com/menglin0320/P5-UIE.git
    ```

1. Download preprocessed data from this [Google Drive link](https://drive.google.com/file/d/1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G/view?usp=sharing), then put them into the *data* folder. If you would like to preprocess your own data, please follow the jupyter notebooks in the *preprocess* folder. Raw data can be downloaded from this [Google Drive link](https://drive.google.com/file/d/1uE-_wpGmIiRLxaIy8wItMspOf5xRNF2O/view?usp=sharing), then put them into the *raw_data* folder.

   
2. Download pretrained checkpoints into *snap* folder. If you would like to train your own P5 models, *snap* folder will also be used to store P5 checkpoints.


3. Pretrain with scripts in *scripts* folder, such as

    ```
    bash scripts/pretrain_P5_small_beauty.sh 4
    ```
    or you can run train.sh to to do end to end training
    ```
    bash scripts/train.sh 4
    ```
   Here *4* means using 4 GPUs to conduct parallel pretraining.
    
4. Evaluate with example jupyter notebooks in the *notebooks* folder. Before testing, create a soft link of *data* folder to the *notebooks* folder by
   
   ```
   cd notebooks
   ln -s ../data .
   ```

## Document for this project
['report'](https://lbuhk29fzve.larksuite.com/docx/WWfZdeM4XoIaAQx3aEBuXoFhszd).

## Pretrained Checkpoints
See [CHECKPOINTS.md](snap/CHECKPOINTS.md).
A google drive link for a checkpoint for P5 with soft embedding is also added to the CHECKPOINTS mark down file.

## Citation

Please cite the following paper for the original work:
```
@inproceedings{geng2022recommendation,
  title={Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt \& Predict Paradigm (P5)},
  author={Geng, Shijie and Liu, Shuchang and Fu, Zuohui and Ge, Yingqiang and Zhang, Yongfeng},
  booktitle={Proceedings of the Sixteenth ACM Conference on Recommender Systems},
  year={2022}
}
```

## Acknowledgements

[VL-T5](https://github.com/j-min/VL-T5), [PETER](https://github.com/lileipisces/PETER), [S3-Rec](https://github.com/aHuiWang/CIKM2020-S3Rec) and [P5 repo](https://github.com/jeykigung/P5.git)
