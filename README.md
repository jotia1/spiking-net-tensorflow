# spiking-net-tensorflow 2.0
This repository has code to run spiking net simulations in Tensorflow 2.0. The code is under active development and the api/functions are prone to change. Feedback and comments are appreciated, contributions are also welcome. 

## Plastic synaptic delays between neurons
There is [some evidence](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6310485/)[1] to suggest delay plasticity is an active part of learning in the brain (much like STDP, but changing delays between neurons instead of weights). This repository differentiates itself from other spiking net simulators by having support for **plastic synaptic delays** between neurons. The implemented delay learning rule is SDVL outlined in [this paper](https://ieeexplore.ieee.org/document/6252371)[2].

## Usage
To use this repository, first install tensorflow 2.0 as per the [tensorflow website](https://www.tensorflow.org/install/pip). Ealier versions of tensorflow will NOT work.

```
pip install tensorflow==2.0.0-alpha0
```

Clone this repository, change into the directory and run an example.

```
git clone https://github.com/jotia1/spiking-net-tensorflow

cd spiking-net-tensorflow

python runnetwork.py
```

A plot of the network should appear.


Key features:
- Plastic delays between neurons
- Variable time step for simulations
- LIF neurons implemented
- STDP functional
- SDVL functional 


A rough roadmap of development:
- Generate examples with delayed LIF neurons
- Add tests
- Add examples 

Initial development was done [here](https://colab.research.google.com/drive/1ompErt-hmnRBYUk6fjcRyqrc8Et2cXxY) and will give some insight into how the simulator works

If you are interested in this space (spiking nets/delays/etc.) you are encouraged to get in contact with the author

References:

[1] Fields R. D. (2015). A new mechanism of nervous system plasticity: activity-dependent myelination. Nature reviews. Neuroscience, 16(12), 756–767. doi:10.1038/nrn4023

[2] P. W. Wright and J. Wiles, “Learning transmission delays in spiking neural networks: A novel approach to sequence learning based on spike delay variance,” in The 2012 International Joint Conference on Neural Networks (IJCNN), 2012, pp. 1–8.
