# spiking-net-tensorflow 2.0
This repository has code to run spiking net simulations in Tensorflow 2.0. The code is under active development and the api/functions are prone to change. Feedback and comments are appreciated, contributions are also welcome. 

## Plastic synaptic delays between neurons
There is [some evidence](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6310485/)[1] to suggest delay plasticity is an active part of learning in the brain (much like STDP, but changing delays between neurons instead of weights). This repository differentiates itself from other spiking net simulators by having support for **plastic synaptic delays** between neurons. The implemented delay learning rule is SDVL outlined in [this paper](https://ieeexplore.ieee.org/document/6252371)[2].


A rough roadmap of development:
- Generate examples with delayed LIF neurons
- Add tests
- Add examples 


If you are interested in this space (spiking nets/delays/etc.) you are encouraged to get in contact with the author

References:
[1] Fields R. D. (2015). A new mechanism of nervous system plasticity: activity-dependent myelination. Nature reviews. Neuroscience, 16(12), 756–767. doi:10.1038/nrn4023
[2] P. W. Wright and J. Wiles, “Learning transmission delays in spiking neural networks: A novel approach to sequence learning based on spike delay variance,” in The 2012 International Joint Conference on Neural Networks (IJCNN), 2012, pp. 1–8.
