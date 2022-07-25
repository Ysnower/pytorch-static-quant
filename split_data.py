from glob import glob
import random
import os

trainval = 0.9
train = 0.9
test = 0.1
for i in range(2):
    sign = str(i)
    img_list = sorted(glob('./data/' + str(i) + '/*'))
    trainvalNum = int(trainval*len(img_list))
    trainval_list = random.sample(img_list, trainvalNum)

    train_list = random.sample(trainval_list, int(len(trainval_list)*train))

    saveBasePath = './'
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'a+')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'a+')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'a+')
    for i in img_list:
        if i in trainval_list:
            if i in train_list:
                ftrain.write(i+' '+sign+'\n')
            else:
                fval.write(i+' '+sign+'\n')
        else:
            ftest.write(i+' '+sign+'\n')