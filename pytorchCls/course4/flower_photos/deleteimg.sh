#!/bin/bash


a=/home/mao/Downloads/datasets/natural-scenesError/street   #文件夹a
b=/home/mao/Downloads/datasets/natural-scenes/seg_train/street   #文件夹b

for i in `ls $a`
do
echo $i
ls -l $b/$i
rm $b/$i
done