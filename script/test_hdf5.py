#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
dataの追加のテスト
"""

import h5py

output_file = "random.h5"
h5file = h5py.File(output_file, 'w')
dirs = h5file.create_group("dir")

for i in range(3):
    if i == 0:
        dirs.create_dataset("test",data= i,chunks=True)
        h5file.flush()
    else:
        pass
    
    
h5file.flush()
h5file.close()
