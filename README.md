# MultiDNN_adaptive_mem_management
Apative Memory Management for Multi DNN Acceelrators

To run baseline:
python3 simulator_mini_batching_baseline.py arch model1 model2 model3 model4 bsize1 basize2 basize3 bsize4

eg:

python3 simulator_mini_batching_baseline.py aimt resnet resnet resnet resnet 8 8 8 8

To run proposed system:

python3 simulator_mini_batching_v2.py arch model1 model2 model3 model4 bsize1 basize2 basize3 bsize4

eg:

python3 simulator_mini_batching_v2.py aimt resnet resnet resnet resnet 8 8 8 8


