# -*- coding: utf-8 -*-
# @Time    : 2022/9/6 15:56
# @Author  : Kenny Zhou
# @FileName: pHash.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com

from pathlib import Path
import os
import shutil
import pickle

# 感知哈希算法
import cv2
import numpy as np
#定义感知哈希
def phash(img):
    #step1：调整大小32x32
    img=cv2.resize(img,(32,32))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=img.astype(np.float32)

    #step2:离散余弦变换
    img=cv2.dct(img)
    img=img[0:8,0:8]
    sum=0.
    hash_str=''

    #step3:计算均值
    # avg = np.sum(img) / 64.0
    for i in range(8):
        for j in range(8):
            sum+=img[i,j]
    avg=sum/64

    #step4:获得哈希
    for i in range(8):
        for j in range(8):
            if img[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

#计算汉明距离
def hmdistance(hash1,hash2):
    num=0
    assert len(hash1)==len(hash2)
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            num+=1
    return num


def cal_hash(path):
    img1=cv2.imread(path)
    hash1=phash(img1)
    return hash1

# 提前计算哈希表
def auto_hash(input_dir):
    input_dir = Path(input_dir)
    hash_dict = {}

    for file_name in input_dir.glob("**/*.jpg"):
        hash_dict[cal_hash(str(file_name))] = str(file_name.absolute())
    with open('pickle.hash', 'wb') as f:
        pickle.dump(hash_dict, f)

def load_hash(input_path):
    input_path = Path(input_path)
    with open(input_path.absolute(), 'rb') as f:
        hash_dict = pickle.load(f)
    return hash_dict

def map_hash(input,hash_path,N=5):
    hash_dict = load_hash(hash_path)
    all_distance = {}
    input_hash = cal_hash(input)
    for hash in hash_dict.keys():
        all_distance[hash]=hmdistance(hash,input_hash)
    top_list = sorted(all_distance.items(), key=lambda kv: (kv[1], kv[0]))
    top_hash = top_list[:N]
    for i in top_hash:
        print(hash_dict[i[0]],i[1])

def cal_two_imgs():
    img1=cv2.imread('/Volumes/Sandi/Jewelry/R/DRG1596R01WM18WA.jpg')
    img2=cv2.imread('/Volumes/Sandi/Jewelry/R/DRG1819R01W18WA.jpg')

    hash1=phash(img1)
    hash2=phash(img2)

    print(hash1)
    print(hash2)

    dist=hmdistance(hash1,hash2)
    print('距离为：',dist)


if __name__ == '__main__':
    # auto_hash("/Volumes/Sandi/Jewelry/R")
    map_hash("/Volumes/Sandi/Jewelry/R/DRE0219R01W18WRA.jpg","/Users/kennymccormick/github/SPQ/pickle.hash",5)
