# pytorch-static-quant
post training static quantization

##split_data
data目录下存放分类图像数据，这里是2分类所以有data/0和data/1,更多分类继续往后面加，每个文件夹放好对应类别的数据图像，执行split_data.py
会生成train.txt, val.txt, test.txt(推理阶段测试数据)  txt里面文件格式是 imagepath label 

##train
执行train.py 训练好的模型在checkpoint文件夹里面

##quantization
执行fx_ptq.py 可以得到量化后的模型

##predict
推理阶段执行PTQpredict.py 可以加载test.txt读取测试图像，输出结果与label比较计算准确性
