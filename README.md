# BertGAT-for-Spider-Dataset

We use Spider dataset as our main dataset to fulfill the largescale and cross-domain semantic parsing text-to-SQL 
tasks. For this task, we propose BertGAT, which is a novel approach to the beforementioned task. 
To build this model, we implement Bidirectional Encoder Representations from Transformers (BERT) 
to pre-train deep bidirectional representations instead of the traditional Bidirectional recurrent
neural networks. Fine-tuning is applied to the pre-trained BERT representations so that we can
use just one extra output layer to create state-of-the-art models for wide-ranging text-to-SQL tasks.
We use Syntax tree network to employ a treebased SQL generator, and use Graph Attention
networks (GATs) to learn the features of syntax-tree.
