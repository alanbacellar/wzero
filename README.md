# wzero
Weightless Neural Networks Library
  - A Collecation of Weghtless models implementd in in paralellized C++ code with a python binding.
  - A Collectioins of Datasets commonly used in Weightless models
  - and many more.
  
 Mnist Example:
 ```python
 import wzero as wz
 
 dataset = wz.datasets.Mnist().distrib_thermometer(10).flatten()
 model = wz.models.WiSARD2(7840, 40, 10)
 data = wz.experiments.profile(dataset, model)
 
 print(data)
 ```
