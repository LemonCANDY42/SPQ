# -*- coding: utf-8 -*-
# @Time    : 2022/9/6 14:38
# @Author  : Kenny Zhou
# @FileName: paser_class.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com
from pathlib import Path
import os
import shutil

def auto_split_class(input_dir,out_dir):
	input_dir = Path(input_dir)
	out_dir = Path(out_dir)
	class_list = []
	for file_name in input_dir.glob("**/*.jpg"):
		if file_name.stem[1] not in class_list:
			class_list.append(file_name.stem[1])
			Path(out_dir / f'{file_name.stem[1]}').mkdir(parents=True, exist_ok=True)
		folder = Path(out_dir / f'{file_name.stem[1]}')
		shutil.copy(file_name, folder / file_name.name)

if __name__ == '__main__':
	auto_split_class("/Volumes/Sandi/noble-Images-com20220905","/Volumes/Sandi/Jewelry")