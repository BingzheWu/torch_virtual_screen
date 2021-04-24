# Introduction
In most of the open-public molecular property prediction benchmarks (e.g., benchmarks in 
MolecularNet), one molecular graph is always marked with multiple labels. Therefore, multi-label learning can be used for reducing the training overhead also improving training performance. 

## Multi-label learning
## Torch implementation
In Torch framework, `nn.BCEWithLogitsLoss` is devised for multi-label binary classification.

## Loss criterion in Torch
Commonly-used loss criterions in torch include:
+ `nn.BCEWithLogitsLoss` is a combination with `nn.BCELoss` and `nn.Sigmoid`. It is used for binary classification tasks and supports multi-label tasks.
+ `nn.CrossEntropyLoss` is a combination with `nn.NLLloss` and `nn.SoftMax`. It is used for multi-class tasks.