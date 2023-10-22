# Residual Boosting Network (RBN) (ICIEA 2023) 

V.K.Z. Koh, X.Li, Z.Lin, Y.Li, E.Shafiee, and B.Wen, “Deterministic to probabilistic time series forecasting by recurrent neural network with variational residual boosting,” in 2023 IEEE 18th Conference on Industrial Electronics and Applications (ICIEA), pp. 203–208, 2023. 

Residual Boosting Network (RBN) is used to convert deterministic time series forecasts into proabilistic by incropating a novel output layer into existing models. [Paper](https://ieeexplore.ieee.org/abstract/document/10241877)

## Abstract
Classic methods for time series forecasting problems are mostly based on the statistical approach, which is more interpretable and data-efficient. More recently, learning-based methods have demonstrated superior results by adapting better to time series data distribution. However, models mostly forecast deterministic results that do not show the confidence intervals to address the uncertainty. This can be addressed probabilistically to quantify the confidence in the forecasts that are well off for critical decision-making. The paper aims to leverage existing deterministic Recurrent Neural Network (RNN) models to produce probabilistic forecasts. We proposed 
(1) a novel model called the Residual Boosting Network (RBN) that utilizes the backcast residuals obtained from the RNN to exploit the local uncaptured characteristics and provide a correction to the final forecast. 
(2) An estimator layer is introduced to both RNN and RBN to estimate the final probabilistic forecasts. 
(3) A loss function that consists of a reconstruction and correlation regularization term is proposed to ensure RNN and RBN learn different characteristics of the data. The experimental results have shown that utilizing the RBN model produced better probabilistic forecast as compared to other models across different datasets.

# Acknowledgement
We would like to acknowledge the following papers, github repositories and datasets:

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

These have provided us valuable resources to conduct our research
