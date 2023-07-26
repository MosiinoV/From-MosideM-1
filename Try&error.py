# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 08:24:10 2020

@author: mohse
"""
import numpy as np
import matplotlib.pyplot as plt

# %%
A = np.array([[1, 2, 3], [4, 5, 3], [7, 8, 5]])
# B = np.delete(A,0,0)
r, c = np.where(A == 3)
r += 1
sarr = [str(a) for a in r]
if len(r) != 0:
    print("The", "\033[1;32;41m", "G", (", ".join(sarr)), "\033[0m",
          "satellite has error on clock bias and it can be removed from calculations!")

for i in range(3):
    print("OK")

B = [[1, 2, 3], [4, 5, 3], [7, 8, 5]]
# add a column to T, at the front:
F = np.insert(A, 3, 2, axis=1)

a = np.array([[1, 1], [2, 2], [3, 3]])
b = np.insert(a, 1, 5)
d = np.insert(a, 1, 5, axis=1)

n = np.array([[21, 20, 19, 18, 17], [16, 15, 14, 13, 12], [11, 10, 9, 8, 7], [6, 5, 4, 3, 2]])
y = np.argsort(n[:, 2], kind='mergesort')  # a[:,2]=[19,14,9,4]
n = n[y]

x = np.array([['140', 'GGC'], ['256', 'AGGG'], ['841', 'CA'], ['46', 'TTATAGG'],
              ['64', 'AGAGAAAGGATTATG'], ['156', 'AGC'], ['187', 'GGA'], ['701', 'TTCG'],
              ['700', 'TC']])
x = x[x[:, 0].astype(int).argsort()]

# %%
r = np.arange(0, 2, 0.01)
theta = 2 * np.pi * r

# plt.figure(num=1)
ax = plt.subplot(111, projection='polar')
ax.plot(theta, r)
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
ax.set_rlabel_position(-30.5)  # Move radial labels away from plotted line
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')
plt.show()

# %%
r = np.arange(0, 2, 0.01)
theta = 2 * np.pi * r

plt.figure(num=2)
plt.polar(theta, r)
