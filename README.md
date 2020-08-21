# BertGAT-for-Spider-Dataset

Chang Shu, Ruitao Yi, Bo Lun 

We use Spider dataset as our main dataset to fulfill the largescale and cross-domain semantic parsing text-to-SQL 
tasks. For this task, we propose BertGAT, which is a novel approach to the beforementioned task. 
To build this model, we implement Bidirectional Encoder Representations from Transformers (BERT) 
to pre-train deep bidirectional representations instead of the traditional Bidirectional recurrent
neural networks. Fine-tuning is applied to the pre-trained BERT representations so that we can
use just one extra output layer to create state-of-the-art models for wide-ranging text-to-SQL tasks.
We use Syntax tree network to employ a treebased SQL generator, and use Graph Attention
networks (GATs) to learn the features of syntax-tree.

#### Environment Setup

1. The code uses Python 3.6, DGL 0.5.0 and [Pytorch 1.4.0](https://pytorch.org/previous-versions/) GPU.
2. Install Python dependency: `pip install -r requirements.txt`

#### Download Data, Embeddings, Scripts, and Pretrained Models
1. Download the dataset from [the Spider task website](https://yale-lily.github.io/spider) to be updated, and put `tables.json`, `train.json`, and `dev.json` under `data/` directory.
2. Download the pretrained [Glove](https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip), and put it as `glove/glove.%dB.%dd.txt`
3. Download `evaluation.py` and `process_sql.py` from [the Spider github page](https://github.com/taoyds/spider)
4. Download preprocessed train/dev datasets and pretrained models from [here](https://drive.google.com/file/d/1FHEcceYuf__PLhtD5QzJvexM7SNGnoBu/view?usp=sharing). It contains: 
   -`generated_datasets/`
    - ``generated_data`` for original Spider training datasets, pretrained models can be found at `generated_data/saved_models`
    - ``generated_data_augment`` for original Spider + augmented training datasets, pretrained models can be found at `generated_data_augment/saved_models`

#### Generating Train/dev Data for Modules
You could find preprocessed train/dev data in ``generated_datasets/``.

To generate them by yourself, update dirs under `TODO` in `preprocess_train_dev_data.py`, and run the following command to generate training files for each module:
```
python preprocess_train_dev_data.py train|dev
```

#### Folder/File Description
- ``data/`` contains raw train/dev/test data and table file
- ``generated_datasets/`` described as above
- ``models/`` contains the code for each module.
- ``evaluation.py`` is for evaluation. It uses ``process_sql.py``.
- ``train.py`` is the main file for training. Use ``train_all.sh`` to train all the modules (see below).
- ``test.py`` is the main file for testing. It uses ``supermodel.sh`` to call the trained modules and generate SQL queries. In practice, and use ``test_gen.sh`` to generate SQL queries.
- `generate_wikisql_augment.py` for cross-domain data augmentation


#### Training
Run ``train_all.sh`` to train all the modules.
It looks like:
```
python train.py \
    --data_root       path/to/generated_data \
    --save_dir        path/to/save/trained/module \
    --history_type    full|no \
    --table_type      std|no \
    --train_component <module_name> \
    --epoch           <num_of_epochs>
```

#### Testing
Run ``test_gen.sh`` to generate SQL queries.
``test_gen.sh`` looks like:
```
SAVE_PATH=generated_datasets/generated_data/saved_models_hs=full_tbl=std
python test.py \
    --test_data_path  path/to/raw/test/data \
    --models          path/to/trained/module \
    --output_path     path/to/print/generated/SQL \
    --history_type    full|no \
    --table_type      std|no \
```

#### Evaluation
Follow the general evaluation process in [the Spider github page](https://github.com/taoyds/spider).
