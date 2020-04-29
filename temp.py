#!/usr/bin/env python 
# -*- coding:utf-8 -*-
__author__ = 'CheXinkai'
import matplotlib.pyplot as plt

x = [0,5,10,15,20,25]
y1 = [0.94,0.94,0.92,0.89,0.82,0.77]
y2 = [0.206,0.223,0.298,0.428,0.588,0.778]

plt.figure()
plt.plot(x, y1)
plt.xlabel('num of sybil')
plt.ylabel('loss')
plt.savefig('.//log//loss_all.png')

plt.figure()
plt.plot(x, y2)
plt.xlabel('num of sybil')
plt.ylabel('loss')
plt.savefig('.//log//acc_all.png')