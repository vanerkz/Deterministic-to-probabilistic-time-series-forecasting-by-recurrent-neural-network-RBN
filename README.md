# Residual Boosting Network (RBN) (ICIEA 2023) Deterministic to probabilistic time series forecasting by recurrent neural network with variational residual boosting

V.K.Z. Koh, X.Li, Z.Lin, Y.Li, E.Shafiee, and B.Wen, “Deterministic to probabilistic time series forecasting by recurrent neural network with variational residual boosting,” in 2023 IEEE 18th Conference on Industrial Electronics and Applications (ICIEA), pp. 203–208, 2023. 

Residual Boosting Network (RBN) converts deterministic time series forecasts into probabilistic by incorporating a novel output layer into existing models. [Paper](https://ieeexplore.ieee.org/abstract/document/10241877)

## Abstract
Classic methods for time series forecasting problems are mostly based on the statistical approach, which is more interpretable and data-efficient. More recently, learning-based methods have demonstrated superior results by adapting better to time series data distribution. However, models mostly forecast deterministic results that do not show the confidence intervals to address the uncertainty. This can be addressed probabilistically to quantify the confidence in the forecasts that are well off for critical decision-making. The paper aims to leverage existing deterministic Recurrent Neural Network (RNN) models to produce probabilistic forecasts. We proposed 
1. A novel model called the Residual Boosting Network (RBN) utilizes the backcast residuals obtained from the RNN to exploit the local uncaptured characteristics and correct the final forecast.
2. An estimator layer is introduced to both RNN and RBN to estimate the final probabilistic forecasts. 
3. A loss function that consists of a reconstruction and correlation regularization term is proposed to ensure RNN and RBN learn different characteristics of the data. The experimental results have shown that utilizing the RBN model produced better probabilistic forecasts than other models across different datasets.

If required, due to file size, I am unable to upload the electricity dataset. Please download from [link](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) and store the file name as "LD2011_2014.txt" in directory ..\data\elect

## Contact
Please feel free to contact me !!
## Acknowledgement
We want to acknowledge the following papers, github repositories and datasets:

https://github.com/apsarath/NRU

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

These have provided us valuable resources to conduct our research. Deeply Appreciated
