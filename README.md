# ProConDual: Class-Prototype based Contrastive learning with promoting Self-Duality for imbalanced image classification

This is a Pytorch implementation of the [ProConDual paper]():

<p align="center">
<img src="./Arch.png" width="800">
</p>


## Requirement
- pytorch>=2.1.0
- torchvision
- tensorboardX

### CIFAR-10-LT
To do supervised training with ProConDual for 200 epochs on CIFAR-10-LT, run
```
python main_cifar.py --dataset cifar-10-lt
```
(All default parameter setups are included in main_cifar.py)


To evaluate the performance on the test set, run
```
python main_cifar.py --dataset cifar-10-lt --reload True \
  --resume {model_save_path}
```

| Method | IF | Top-1 Acc(%) |
| :---:| :---:|:---:|
| ProConDual   | 100   | 94.47%    |
| ProConDual   | 50    | 94.49%    |
| ProConDual   | 10    | 95.69%    |


### CIFAR-100-LT
To do supervised training with ProConDual for 200 epochs on CIFAR-100-LT, run
```
python main_cifar.py --dataset cifar-100-lt
```
(All default parameter setups are included in main_cifar.py)


To evaluate the performance on the test set, run
```
python main_cifar.py --dataset cifar-100-lt --reload True \
  --resume {model_save_path}
```

| Method | IF | Top-1 Acc(%) |
| :---:| :---:|:---:|
| ProConDual   | 100   | 74.80%    |
| ProConDual   | 50    | 74.90%    |
| ProConDual   | 10    | 75.98%    |



### ImageNet-LT 
To do supervised training with ProConDual for 180 epochs on ImageNet-LT, run
```
python main_imagenet.py
```
(All default parameter setups are included in main_imagenet.py)


To evaluate the performance on the test set, run
```
python main_imagenet.py  --reload True \
  --resume {model_save_path}
```

| Method | Model | Many | Med | Few | All | model |
| :---:| :---:|:---:|:---:|:---:| :---:|  :---:| 
| ProConDual |ResNeXt-50 | 67.6  | 54.1  | 40.0     | 57.4    | [Download]() |


### ISIC2019 
To do supervised training with ProConDual for 600 epochs on ISIC 2019, run
```
python main_ISIC.py
```
(All default parameter setups are included in main_ISIC.py)

To evaluate the performance on the test set, run
```
python main_ISIC.py  --reload True \
  --resume {model_save_path}
```



| Method | Model | Top-1 Acc(%) | link | 
| :---: | :---: | :---: | :---: | 
|ProConDual | ResNet-50   | 87.18 | [Download]() | 

### Datasets txt file

````
ProConDual-main/dataset
├── ImageNet+LT
|   └── ImageNet_LT_test.txt
|   └── ImageNet_LT_train.txt
|   └── ImageNet_LT_val.txt
└── ISIC2019
    └── ISIC2019_test.csv
    └── ISIC2019_train.csv
    └── ISIC2019_val.csv
````

    
