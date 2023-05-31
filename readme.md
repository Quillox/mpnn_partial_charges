# Message-passing Neural Networks (MPNN) for predicting molecular properties

## Introduction

This repository contains the code for training and evaluating message passing neural networks (MPNNs) for predicting molecular properties. 

## Data

The included data is a subset from [Sereina Riniker](https://doi.org/10.3929/ethz-b-000230799).


## Usage

The notebook `run_and_evaluate.ipynb` contains the code for training and evaluating the MPNN.
## Requirements

The required python packages are

```
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install dgllife
pip install torch torchvision torchaudio
pip install rdkit
```

## References

[Machine Learning of Partial Charges Derived from High-Quality Quantum-Mechanical Calculations](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00663)

[Deep mind](https://github.com/deepmind/graph_nets)

[Deep Graph Library](https://www.dgl.ai/)