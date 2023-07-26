# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:11:24 2020

@author: mohsen 
The first part is not usable but the second part is complete :)
It is HIDEN :|
"""
"""
#%% 
import numpy as np
import json
import ast

gps_data3 = []
j = 0
with open("test.txt","r") as f:
    data = f.readlines()
    
A = data[22:]
B = np.array(A)
#C = np.zeros(shape=(len(B),6))
for i in range(len(B)):
    C = A[i][2:]
   
for i in data:
    gps_data3[j,:] = i[1:]
    j+=1
   
#    gps_data2 = json.loads(f)

gps_data1 = data[22:]
#gps_data1 = np.array(data[22:])
for i in range(len(gps_data1)):
    gps_data1[i][0] = []

for i in range(len(gps_data1)):
    gps_data1[i,:] = gps_data1[i,1:]
#gps_data1[:][0] = []

#gps_data2 = json.loads(data)
#gps_data2 = ast.literal_eval(gps_data1)

"""
#%%
import numpy as np

lines1 = [line.split() for line in open('02o.txt')]
lines2 = [line.split() for line in open('sp3.txt')]
List1 = ["02","1","10","0","15","{:<09}".format(0.0)]
List2 = ["2002","1","10","0","15","{:<010}".format(0.0)]

for i in range(len(lines1)):
    if set(["APPROX"]).intersection(set(lines1[i])):
        approx_coord = lines1[i][:3]
    if set(List1).issubset(set(lines1[i])):
        index = i
        satellites_detect = "".join(lines1[i][6:])
        
satellites_str = "".join((ch if ch in '0123456789.-e' else ' ') for ch in satellites_detect)
satellites = [str(k) for k in satellites_str.split()][1:]
pseu_code = lines1[index+1:index+len(satellites)+1][:]

for j in range(len(lines2)):
    if set(List2).issubset(set(lines2[j])):
        coord = lines2[j+1:j+int(lines2[2][1])+1]

y = 0
satellites_coord = np.zeros(shape=(len(satellites),5))
for x in range(len(coord)):
    if set(satellites).intersection(set(coord[x])):
        satellites_coord[y][0] = coord[x][1]
        satellites_coord[y][1:] = coord[x][2:]
        y+=1
        
        
        
        
        
        
        
        
        
        
        