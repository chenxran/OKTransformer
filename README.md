# OK-Transformer

This is the official implementation of

1. Free Lunch for Efficient Textual Commonsense Integration in Language Models (ACL 2023)
2. Enhancing Natural Language Representation with Large-Scale Out-of-Domain Commonsense (ACL 2022 Findings)


To run OK-Transformer, you can directly run the ```main.py``` script. ```--fast=ours``` is the faster implementation of OK-Transformer incoporated well-designed batch partitioning algorithm presented in our ACL 2023 paper.

To directly applying our batch paritioning algorithm, please refer  to ```partition(ex_vi, batch_size)``` function in  ```cluster.py```. Here ```ex_vi``` is a list of all one-hot embedding of knowledges in the training set.