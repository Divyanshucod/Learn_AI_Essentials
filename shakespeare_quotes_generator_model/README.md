## General Process of model training
## Raw text → tokens → batches → model → loss → backprop → optimizer → validation → generation
## Via using only embedding matrix in biagram post some iterations of training, we were able to achieve the loss = 2.394
## In One headed attention loss = 2.35
## In multi headed attention loss = 2.2704
## After adding feedward network with one layer loss = 2.2046
## After Adding residual connections and updates in feedforward layer loss = 2.1425
## After adding layer norm loss = 2.1371