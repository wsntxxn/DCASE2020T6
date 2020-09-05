# DCASE2020 Task 6: Automated Audio Captioning

This repository provides source code for DCASE2020 task 6 [SJTU submission](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Xu_43_t6.pdf).

# Dataset

The dataset used for task 6 is [Clotho](https://arxiv.org/abs/1910.09387). To use our pretrained model, the vocabulary file is provided in `data/clotho/vocab.pth`.

For convenience, we rename all the filenames. The comparison between original and new filenames can be found in `data/clotho/filename.csv`. Original caption label files are also reformatted to json files `data/clotho/dev.json` and `data/clotho/eval.json`.

# Prequisite Installation

First checkout this repository:
```bash
git clone https://github.com/wsntxxn/DCASE2020T6.git --recursive
```

The code is written exclusively in Python3. In order to install all required packages use the included `requirements.txt`. `pip install -r requirements.txt` does the job.

## Extract Features

We save all the keys of the features in a script file (just like kaldi scp) when we extract features. For example, assume the raw wave data are placed in `DATA_DIR` (`data/clotho/wav` here) and you will store features in `FEATURE_DIR` (`data/clotho` here):

```bash
DATA_DIR=`pwd`/data/clotho/wav
FEATURE_DIR=`pwd`/data/clotho
python utils/featextract.py `find $DATA_DIR -maxdepth 1 -type f` $FEATURE_DIR/lomgel.hdf5 $FEATURE_DIR/lomgel.scp mfcc -win_length 1764 -hop_length 882 -n_mels 64
```

The scp file can be further split into a development scp and an evaluation scp according to development-evaluation setting:
```bash
python utils/split_scp.py $FEATURE_DIR/logmel.scp $FEATURE_DIR/eval.json
```

## Training Configurator

Training configuration is done in `config/*.yaml`. Here one can adjust some hyperparameters e.g., number of hidden layers or embedding size. You can also write your own models in `models/*.py` and adjust the config to use that model (e.g. `encoder: MYMODEL`). 

Note: All parameters within the `runners/*.py` script use exclusively parameters with the same name as their `.yaml` file counterpart. They can all be switched and changed on the fly by passing `--ARG VALUE`, e.g., if you wishes to train for 40 epochs, pass `--epochs 40`.


### Cross Entropy Training

In order to train a model using standard cross entropy loss, run:

```bash
python runners/run.py train config/xe.yaml
```

### Self-Critical Sequence Training (SCST)

To train a model using SCST, run:

```bash
python runners/run_scst.py train config/scst.yaml
```

Remember to specify `pretrained` with cross entropy trained model as the starting model, otherwise it is hard for SCST to work properly.

Training logs and model checkpoints in `OUTPUTPATH/MODEL/TIMESTAMP`.

## Predicting and Evaluating

Predicting and evaluating is done by running `evaluate` function in `runners/*.py` (assume the evaluation labels are in `data/clotho/eval.json`):

We provide submission model weights: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4013181.svg)](https://doi.org/10.5281/zenodo.4013181).

For example, after download `DCASE2020_submission1.pth`, to use it for prediction and evaluation:
```bash
EXP_PATH=experiments/CaptionModel/DCASE2020_submission1
mkdir -p $EXP_PATH
mv DCASE2020_submission1.pth $EXP_PATH/saved.pth
python runners/run.py evaluate $EXP_PATH data/clotho/logmel.hdf5 data/clotho/logmel_eval.scp data/clotho/eval.json
```

Standard machine translation metrics (BLEU@1-4, ROUGE-L, CIDEr, METEOR and SPICE) are included. The scores can be seen in `EXP_PATH/scores.txt` while outputs on the evaluation data is in `EXP_PATH/eval_output.json`.

