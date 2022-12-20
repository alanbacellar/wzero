# wzero
Weightless Neural Networks Library
  - A Collecation of Weghtless models implementd in in paralellized C++ code with a python binding.
    - WiSARD
    - Bleaching WiSARD
    - Bloom WiSARD
    - Bleaching Bloom WiSARD
    - PseudoConvWiSARD
    - etc
  - A Collectioins of Datasets commonly used in Weightless models
  - and many more.
 
 Compilation:
 ```
 cd wzero/models/wrappers
 cmake CMakeLists.txt
 make
 ```
 
 Mnist Example:
 ```python
 import wzero as wz
 
 dataset = wz.datasets.Mnist().distrib_thermometer(10).flatten()
 model = wz.models.WiSARD2(7840, 40, 10)
 data = wz.experiments2.profile(dataset, model)
 
 print(data)
 ```
