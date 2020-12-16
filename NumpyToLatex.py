#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 01:25:38 2020

@author: shri
"""
import numpy as np

def np2lat(A):
	filename = 'table.txt'
	f = open(filename, 'a')
	cols = A.shape[1]

	# Change alignment and format of your output
	tabformat = '%.5f'
	tabalign = 'c'*cols

	f.write('\n\\begin{table}[h]\n')
	f.write('\\centering\n')
	f.write('\\begin{tabular}{%s}\n' %tabalign)

	# Use some numpy magic, just addding correct delimiter and newlines
	np.savetxt(f, A, fmt=tabformat, delimiter='\t&\t', newline='\t \\\\ \n')

	f.write('\\end{tabular}\n')
	f.write('\\end{table}\n')

	f.flush()
	f.close()