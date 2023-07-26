# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:51:51 2020

@author: mohsen feizabadi
"""
import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt

print("****************************************************")
print("POINT POSITIONING BASED ON PSEUDORANGE AND GEOMETRY DILUTION OF PRECISION (GDOP)")
print("****************************************************")
"""
------------------------------------------------------------
SECTION 1: IMPORT DATA
------------------------------------------------------------
In this part the interseted date of observation is taken from keyboard. The 
format of date must be written like;  yyyy,mm,dd,h,m
    1- It is not need the ',' between values.
    2- The epoch has recorded in zero second (0.000) in .sp3 file which will 
        be added to date, automatically.
    3- The one digit numbers should be written like normal shape and they 
        should not start with zero (like 03) 
If there is an error to write the date, the following codes warn it to user.
    1- The year must be smaller than 2020.
    2- In .sp3 file, the observations record in each 15 minute. Therefore the
        input minute should be a multiple of 15.
    3- Warning or mistakes are shown in different colors.
"""
Loop = 0
while Loop == 0:
    required_date = [int(required_date) for required_date in input(
        "Enter the date and time of epoch (yyyy,mm,dd,h,m). Note: the epoch has recorded in zero second (0.000):\n").split()]
    if required_date[0] > 2020:
        print("Warning: The year should not be bigger than 2020. You wrote", "\033[1;31;42m", required_date[0],
              "\033[0m")
    elif required_date[1] > 12:
        print("Warning: The month is not accepted. You wrote", "\033[1;31;42m", required_date[1], "\033[0m")
    elif required_date[2] > 31:
        print("Warning: The day is not accepted. You wrote", "\033[1;31;42m", required_date[2], "\033[0m")
    elif required_date[3] > 24:
        print("Warning: The hour should be between 0 to 24. You wrote", "\033[1;31;42m", required_date[3], "\033[0m")
    elif required_date[4] % 15 != 0:
        print("Warning: The minute should be selected as 0, 15, 30 or 45. You wrote", "\033[1;31;42m", required_date[4],
              "\033[0m")
    else:
        Loop = 1
"""
In this part the input date converts to string with some modifications
    1- Date format is different in .sp3 and .02o files. Therefore, for each
        specific variable was considered.
        a- The year value in .02o file writes in two digit yy and in .sp3 file
            writes in four digit yyyy
        b- The second value in .sp3 has 8 decimal but it has 7 decimal in .02o
            files.
The methods of the following code relate to list. 
Note: in receive date and detect the required values from files, we works with 
    the lists of strings.
"""
A = str(required_date[0] % 2000).zfill(2)
required_date_sp3 = [str(i) for i in required_date]
required_date_sp3 += ["{:<010}".format(0.0)]
required_date_02o = required_date_sp3.copy()
required_date_02o.pop(0)
required_date_02o.insert(0, A)
required_date_02o.pop(-1)
required_date_02o.append("{:<09}".format(0.0))
"""
In this part the required files (.sp3 and .02o) insert to program by windows
    dialoge.
"""
print("---------------------------------------------------")
print("Select satellite position (.sp3) file")
print("---------------------------------------------------")
input_file_sp3 = filedialog.askopenfilename(title="Select satellite position (.sp3) file")
print("---------------------------------------------------")
print("Select approximation coordinate (.02o) file")
print("---------------------------------------------------", "\n")
input_file_02o = filedialog.askopenfilename(title="Select approximation coordinate (.02o) file")
"""
In this part all values of each file (.sp3 and .02o) read line by line and
    they assign to a specific variable.
"""
read_file_02o = [line.split() for line in open(input_file_02o, "r")]
read_file_sp3 = [line.split() for line in open(input_file_sp3, "r")]
"""
In this part;
    1- The approxomation coordinates of receiver is detected from .02o file. 
    2- The inserted date (from keyboard) is compared with each line of .02o
        file and the specifications of the satellites in this epoch is 
        extracted.
        a- In order to take the pseudorange codes (L1, L2, C, P1 and P2),
            a specific index is considered (index=i) 
"""
for i in range(len(read_file_02o)):
    if {"APPROX"}.intersection(set(read_file_02o[i])):
        approx_coord = read_file_02o[i][:3]
    if required_date_02o == read_file_02o[i][:6]:
        index = i
        satellites_detect = "".join(read_file_02o[i][6:])
"""
In this part the numbers of satellites and their pseudorange codes is detected.
"""
satellites_str = "".join((ch if ch in '0123456789.-e' else ' ') for ch in satellites_detect)
satellites = [str(k) for k in satellites_str.split()][1:]
pseu_code = read_file_02o[index + 1:index + len(satellites) + 1][:]
pseu_code = np.insert(pseu_code, 0, satellites, axis=1)
pseu_code = pseu_code[pseu_code[:, 0].astype(int).argsort()]
"""
In this part the coordinates of the all satelltes in considered epoch is 
    extracted from .sp3 file.
Note: the date format is different in .sp3 fil
"""
for j in range(len(read_file_sp3)):
    if required_date_sp3 == read_file_sp3[j][1:]:
        coord = read_file_sp3[j + 1:j + int(read_file_sp3[2][1]) + 1]
"""
In this part coordinates of the detected satellites from .02o file is created
    from .sp3 file.
Note: set(a).intersection(set(b)) function is used to find the exact satellite.
"""
y = 0
satellites_coord = np.zeros(shape=(len(satellites), 5))
for x in range(len(coord)):
    if set(satellites).intersection(set(coord[x])):
        satellites_coord[y][0] = coord[x][1]
        satellites_coord[y][1:] = coord[x][2:]
        y += 1
"""
If there is an error in recorded clock bias of any satellite, this satellite 
    must be removed from calculations.
In this part, firstly the clock bias error is detected and after that, it is 
    deleted.
"""
[r, c] = np.where(satellites_coord == 999999.999999)
if len(r) != 0:
    print("The satellite", "\033[1;32;41m", ', '.join(map(str, satellites_coord[r, 0].astype(int))), "\033[0m",
          "has error on clock bias and it is removed from calculations!")
satellites_coord = np.delete(satellites_coord, r, 0)
pseu_code = np.delete(pseu_code, r, 0)
"""
-----------------------------------------------------------
SECTION 2: ANALYSIS DATA
-----------------------------------------------------------
In this part distance between satellite and antenna is calcualted (di).
Note: Unit of Satellite coordinates are 'Km' and unit of Satellite clock
    is 'Microsec'
             _______________________________
            /
    di =   / (Xi-X)^2 + (Yi-Y)^2 + (Zi-Z)^2
         \/
"""
light_speed = 299792458
approx_coord = [float(i) for i in approx_coord]
num_sat = len(satellites_coord)
dx = np.zeros(shape=(num_sat, 1))
dy = np.zeros(shape=(num_sat, 1))
dz = np.zeros(shape=(num_sat, 1))
di = np.zeros(shape=(num_sat, 1))
dXi = np.zeros(shape=(num_sat, 3))
for i in range(num_sat):
    dx[i] = (satellites_coord[i, 1] * 1000) - approx_coord[0]
    dy[i] = (satellites_coord[i, 2] * 1000) - approx_coord[1]
    dz[i] = (satellites_coord[i, 3] * 1000) - approx_coord[2]
    di[i] = np.sqrt(np.power(dx[i], 2) + np.power(dy[i], 2) + np.power(dz[i], 2))
    dXi[i] = [dx[i], dy[i], dz[i]]
"""
Adjustment procedure:        V = JX - L
    J = Coefficient matrix (Jacobian)
    X = unKnown parametres matrix (delta(x))
    L = Observation matrix (Distance observation - Distance calculation)
    V = Residual matrix
Calculation of "J" matrix for at least 4 satelliet based of Taylor series:
        |(x1-x0)/d1   (y1-y0)/d1   (z1-z0)/d1  -light_speed|
    J = |(x2-x0)/d2   (y2-y0)/d2   (z2-z0)/d2  -light_speed|
        |(x3-x0)/d3   (y3-y0)/d3   (z3-z0)/d3  -light_speed|
        |(x4-x0)/d4   (y4-y0)/d4   (z4-z0)/d4  -light_speed|
"""
Jacobian = np.zeros(shape=(num_sat, 4))
for i in range(num_sat):
    for j in range(3):
        Jacobian[i, j] = -dXi[i][j] / di[i]
    Jacobian[i, 3] = -(light_speed * 10 ** -9)
"""
Calculation of "L" matrix for at least 4 satelliet:
    |d1_observed(C1,P1 or P2) - d1_calculated - (light_speed * clock_bias_1)|
L = |d2_observed(C1,P1 or P2) - d2_calculated - (light_speed * clock_bias_2)|
    |d3_observed(C1,P1 or P2) - d3_calculated - (light_speed * clock_bias_3)|
    |d4_observed(C1,P1 or P2) - d4_calculated - (light_speed * clock_bias_4)|
Note: "L" matrix is calculated for all "C1", "P1" and "P2"
"""
L = np.zeros(shape=(num_sat, 3))
for i in range(num_sat):
    for j in range(3):
        L[i, j] = float(pseu_code[i, j + 3]) - di[i] + satellites_coord[i, 4] * light_speed * 10 ** -6
"""
Calculation of "X" matrix for at least 4 satelliet (this matrix actually is 
    the (delta_Xi) matrix):
    X = inv(J' * P * J) * (J' * P * L)
and calculation "Variance Covariance matrix of Unknown parameters"
Note: The weight matrix (P) assumed as unit matrix (I)
"""
X = np.linalg.pinv(Jacobian.T.dot(Jacobian)).dot(Jacobian.T.dot(L))
var_cov = np.linalg.pinv(Jacobian.T.dot(Jacobian))
"""
Calculation of exact point's coordinate:
    X = X0 + delta_x  
    Y = Y0 + delta_y  
    Z = Z0 + delta_z
    Receiver clock bias = T0 + delta_Receiver clock bias  
"""
exact_coord = np.zeros(shape=(3, 3))
for i in range(3):
    for j in range(3):
        exact_coord[i, j] = approx_coord[i] + X[i, j]
adjust_result = np.insert(exact_coord, 3, X[3], axis=0)
"""
Calculation of "Geodetic coordinates" of station.
In this part by using the "ecef2Geodetic" function, this transformation is done.
Note: the formulas are in the following reference:
    C:\\Users\mohse\OneDrive-Universite de Montreal\Codes\Matlab Codes\
    Satellite Geodesy\Homework_pdf
"""
from ecef2Geodetic import ecef2Geodetic

geodetic_coord = np.zeros(shape=(3, 3))
for i in range(3):
    lat, lon, h = ecef2Geodetic(exact_coord[0, i], exact_coord[1, i], exact_coord[2, i])
    geodetic_coord[:, i] = [lat, lon, h]
"""
Calculation of rotation matrix (R) for station:
        |−sin𝜆.cos𝜑  −sin𝜆.sin𝜑  cos𝜆|
    R = |  −sin𝜑       cos𝜑       0 |
        | cos𝜆.cos𝜑   cos𝜆.cos𝜑  sin𝜆|
and calculation of Variance Covariance matrix on Topocentric Frame:
                       |sigma_n^2  sigma_ne   sigma_nu|
    CT = R * CX * R' = |sigma_en   sigma_e^2  sigma_eu|
                       |sigma_un   sigma_ue  sigma_u^2|
"""
R = np.zeros(shape=(9, 3))
for i in range(3):
    R[3 * i, 0] = -np.sin(geodetic_coord[1, i]) * np.cos(geodetic_coord[0, i])
    R[3 * i, 1] = -np.sin(geodetic_coord[1, i]) * np.sin(geodetic_coord[0, i])
    R[3 * i, 2] = np.cos(geodetic_coord[1, i])
    R[3 * i + 1, 0] = -np.sin(geodetic_coord[0, i])
    R[3 * i + 1, 1] = np.cos(geodetic_coord[0, i])
    R[3 * i + 2, 0] = np.cos(geodetic_coord[1, i]) * np.cos(geodetic_coord[0, i])
    R[3 * i + 2, 1] = np.cos(geodetic_coord[1, i]) * np.sin(geodetic_coord[0, i])
    R[3 * i + 2, 2] = np.sin(geodetic_coord[1, i])
var_cov_topocentric = R.dot(var_cov[0:3, 0:3]).dot(R.T)
"""
Calculation of DOP factors: VDOP - HDOP - PDOP - TDOP - GDOP
    VDOP = sigma_u
    HDOP = sqrt(sigma_n^2 + sigma_e^2)
    PDOP = sqrt(sigma_n^2 + sigma_e^2 + sigma_u^2)
    TDOP = sigma_t (this value was calculated in Variance Covariance matrix of 
            Unknown parameters)
    GDOP = sqrt(sigma_n^2 + sigma_e^2 + sigma_u^2 + sigma_t^2)
"""
vdop = np.zeros(shape=(1, 3))
hdop = np.zeros(shape=(1, 3))
pdop = np.zeros(shape=(1, 3))
tdop = np.zeros(shape=(1, 3))
gdop = np.zeros(shape=(1, 3))
for i in range(3):
    vdop[0, i] = np.sqrt(var_cov_topocentric[3 * i + 2, 3 * i + 2])
    hdop[0, i] = np.sqrt(var_cov_topocentric[3 * i, 3 * i] + var_cov_topocentric[3 * i + 1, 3 * i + 1])
    pdop[0, i] = np.sqrt(var_cov_topocentric[3 * i, 3 * i] + var_cov_topocentric[3 * i + 1, 3 * i + 1] + var_cov_topocentric[3 * i + 2, 3 * i + 2])
    tdop[0, i] = np.sqrt(var_cov[3, 3])
    gdop[0, i] = np.sqrt(var_cov_topocentric[3 * i, 3 * i] + var_cov_topocentric[3 * i + 1, 3 * i + 1] + var_cov_topocentric[3 * i + 2, 3 * i + 2] + var_cov[3, 3])
"""
Calculation of ECEF coordinates (X,Y,Z) to Topocentric system (e,n,u)
    |n|       |X|
    |e| = R * |Y|
    |u|       |Z|
"""
dX = np.zeros(shape=(num_sat, 3))
dY = np.zeros(shape=(num_sat, 3))
dZ = np.zeros(shape=(num_sat, 3))
for i in range(num_sat):
    j = 0
    while j < 3:
        dX[i, j] = satellites_coord[i, 1] * 1000 - exact_coord[0, j]
        dY[i, j] = satellites_coord[i, 2] * 1000 - exact_coord[1, j]
        dZ[i, j] = satellites_coord[i, 3] * 1000 - exact_coord[2, j]
        j += 1
topo_coord_C1 = R[:3, :].dot([dX[:, 0], dY[:, 0], dZ[:, 0]])
topo_coord_P1 = R[3:6, :].dot([dX[:, 1], dY[:, 1], dZ[:, 1]])
topo_coord_P2 = R[6:, :].dot([dX[:, 2], dY[:, 2], dZ[:, 2]])
topo_coord = np.concatenate((topo_coord_C1, topo_coord_P1, topo_coord_P2))
"""
% Computes elevation angle, range, and azimuth of satellites and station
"""
range_dist = np.zeros(shape=(num_sat, 3))
elevation_angle = np.zeros(shape=(num_sat, 3))
azimuth = np.zeros(shape=(num_sat, 3))
for i in range(num_sat):
    for j in range(3):
        range_dist[i, j] = np.sqrt(topo_coord[3 * j, i] ** 2 + topo_coord[3 * j + 1, i] ** 2 + topo_coord[3 * j + 2, i] ** 2)
        elevation_angle[i, j] = np.pi / 2 - np.arccos(topo_coord[3 * j + 2, i] / range_dist[i, j])
        azimuth[i, j] = np.arctan(topo_coord[3 * j + 1, i] / topo_coord[3 * j, i])
        if topo_coord[3 * j + 1, i] < 0:
            azimuth[i, j] = azimuth[i, j] + np.pi
        elif topo_coord[3 * j, i] < 0 < topo_coord[3 * j + 1, i]:
            azimuth[i, j] = (2 * np.pi) + azimuth[i, j]
"""
-----------------------------------------------------------
SECTION 3: PLOT POSITIONS
-----------------------------------------------------------
"""
ax = plt.subplot(111, projection='polar')
ax.plot(azimuth[:, 0], range_dist[:, 0] / 2000000, 'or')
ax.set_rmax(15)
ax.set_rticks([3, 6, 9, 12, 15])
ax.set_rlabel_position(-22.5)
ax.grid(True)
for i, j in enumerate(satellites_coord[:, [0]]):
    #  label = f"{(satellites_coord[0,0])}"
    ax.annotate(', '.join(map(str, j.astype(int))), (azimuth[i, 0], range_dist[i, 0] / 2000000),
                textcoords="offset points", xytext=(0, 5), ha='center')
ax.set_title("SATELLITE POSITION", fontdict={"color": "red"}, pad=10)
