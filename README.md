# cnn-classification-dog-vs-cat
基于CNN的图像分类器，使用Kaggle的猫狗图片数据。

- train.py: 自建的简单CNN，训练后测试集精度约83%。
- pre_train.py: 利用已训练的常用网络(基于[ImageNet](http://www.image-net.org/)数据集训练)，进行迁移学习，测试集精度约95%以上。
- data_helper.py: 数据读取和预处理模块。
- img_cnn.py: 基于TensorFlow的自定义简单卷积神经网络。

猫狗图像数据来源：
https://www.kaggle.com/c/dogs-vs-cats/data

keras中载入已训练网络的方法：
https://keras.io/applications/

keras中图像预处理的相关功能介绍：
https://keras.io/preprocessing/image/