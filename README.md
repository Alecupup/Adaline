# Adaline
导入外部数据iris.data，其中包括三种鸢尾花，区间在[0,50]、[51,100]、[101,150]

为了能够学习到全部的样本，应打乱样品的序列，本算法为了方便采用二分类

批量梯度下降进行训练

分别用α=0.01和α=0.0001进行训练，可以看出选对α对训练的重要性

![image](https://user-images.githubusercontent.com/68764044/170650825-779fcc79-c422-4cfa-bda2-642b8af2a8fc.png)

由图看出，若α取值太大可能经过训练误差反而增大

进行特征放缩

![image](https://user-images.githubusercontent.com/68764044/170651697-eaba80ac-e5b5-405b-9e3e-4299b285bd2a.png)
