The PyTorch implementation of gACNN
===============
Pytorch reimplementation of gACNN model that was proposed in the paper [Occlusion Aware Facial Expression Recognition
Using CNN With Attention Mechanism](https://vipl.ict.ac.cn/uploadfile/upload/2019123017182739.pdf) TIP 2019. 

Framework of gACNN model
![](https://github.com/hanluyt/gACNN_pytorch/blob/main/framework.png)

Requirements
------
* Python
* PyTorch
* tqdm

Arguments
------
```
--name             Name of this run
--num_steps        Total number of training epochs to perform  
```


Train Model
-----
```
python main.py --name acnn --batchsize 128 --num_steps 800 --learning_rate 1e-1 --weight_decay 5e-4
```

