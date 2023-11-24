import os, cv2, sys, copy, random
import minutiae_extractor
import numpy as np
import scipy.stats as stats
from math import *
from numpy import fft, linalg
from helper import *
from PIL import Image, ImageDraw
from delaunay_triangulation.triangulate import delaunay
from delaunay_triangulation.typing import Vertex

R = 300 # Max Radial Distance

# PCB Feature Set Params
PCB_STEP_RHO = 5 # RANGE 5 to 20
PCB_STEP_ALPHA = 15 # RANGE 15 to 40 Degrees
PCB_STEP_BETA = 15 # RANGE 15 to 40 Degrees

# DTB Feature Set Params
DTB_STEP_L = 15 # RANGE 15 to 25
DTB_STEP_ALPHA = 15 # RANGE 15 to 20 Degrees
DTB_STEP_O = 15 # RANGE 15 to 20 Degrees

# Feature Decorrelation Algorithm Params
LC = 20000 # Must be less than L*H*S
NS = 1000 # Must divide LC

# Projection Matrix Params
PROJ_MAT_SEED = 1337 # Must be 32-bit integer
PROJ_MAT_Y_DIM = 50 # Must be between 1-LC

# Delaunay Code Permutation Params
PERM_PHI = 200

# Final Score Params
FINAL_RHO = 0.7

# Function for calculating the Feature Code P(mi) based on minutiae point m0
def PCBFeatureCode(minutiae_points, m0):

    # Setting up constants for cube dimensions
    L = floor(R/PCB_STEP_RHO)
    S = floor(360/PCB_STEP_ALPHA)
    H = floor(360/PCB_STEP_BETA)

    cube = [[[0 for i in range(H)] for j in range(S)] for k in range(L)]

    for point in minutiae_points:

        xDiff = point[0] - m0[1]
        yDiff = point[1] - m0[1]

        # Calculate Rho (Radial Distance)
        rho = sqrt(pow(xDiff, 2) + pow(yDiff, 2))

        # Calculate Alpha (Radial Angle)
        alpha = degrees(atan2(yDiff,xDiff))
        if alpha < 0:
            alpha += 360

        # Calculate Beta (Orientation Difference)
        beta = abs(point[2] - m0[2])

        # Quantization of (Rho, Alpha, Beta)
        rho_index = floor(rho/PCB_STEP_RHO)
        if rho_index >= L:
            rho_index = L-1

        alpha_index = floor(alpha/PCB_STEP_ALPHA)
        if floor(alpha/PCB_STEP_ALPHA) >= S:
            alpha_index = S-1

        beta_index = floor(beta/PCB_STEP_BETA)
        if floor(beta/PCB_STEP_BETA) >= H:
            beta_index = H-1

        # The points that count must fall within radius R
        if rho <= R:
            cube[rho_index][alpha_index][beta_index] = 1

    # Break down the cube into a single binary array and output it
    feature_code = []
    for i in cube:
        for j in i:
            for k in j:
                feature_code.append(k)

    return feature_code

# Function for calculating the Delaunay Feature Code
def DTBFeatureCode(triangle):

    # Quantization bitstring lengths
    O_LEN = floor(log2(360/DTB_STEP_O)) + 1
    A_LEN = floor(log2(360/DTB_STEP_ALPHA)) + 1
    L_LEN = floor(log2(R/DTB_STEP_L)) + 1
    T_LEN = 3

    # Edge Lengths
    m1m2 = sqrt(pow(triangle[0][0]-triangle[1][0], 2)
              + pow(triangle[0][1]-triangle[1][1], 2))
    m2m3 = sqrt(pow(triangle[1][0]-triangle[2][0], 2)
              + pow(triangle[1][1]-triangle[2][1], 2))
    m1m3 = sqrt(pow(triangle[0][0]-triangle[2][0], 2)
              + pow(triangle[0][1]-triangle[2][1], 2))

    # Calculating O_CODE
    O = abs(triangle[1][2] - triangle[2][2])
    O = bin(floor(O / DTB_STEP_O))[2:]
    O = ('0' * (O_LEN - len(O) - 1)) + O
    O_CODE = []
    for i in O:
        O_CODE.append(int(i))

    # Calculating A_CODE
    A = degrees(acos((pow(m1m3,2) + pow(m1m2,2) - pow(m2m3,2)) / (2 * m1m3 * m1m2)))
    A = bin(floor(A / DTB_STEP_ALPHA))[2:]
    A = ('0' * (A_LEN - len(A) - 1)) + A
    A_CODE = []
    for i in A:
        A_CODE.append(int(i))

    # Calculating L1_CODE
    L = m1m2
    L = bin(floor(L / DTB_STEP_L))[2:]
    L = ('0' * (L_LEN - len(L) - 1)) + L
    L1_CODE = []
    for i in L:
        L1_CODE.append(int(i))

    # Calculating L2_CODE
    L = m1m3
    L = bin(floor(L / DTB_STEP_L))[2:]
    L = ('0' * (L_LEN - len(L) - 1)) + L
    L2_CODE = []
    for i in L:
        L2_CODE.append(int(i))

    # Concatenate all CODE binary arrays to generate the feature code
    feature_code = [O_CODE, L1_CODE, L2_CODE, A_CODE]
    return feature_code

# Function for permuting the Delaunay Feature code
def PermDelFeatCode(code):
    LD = pow(2, np.shape(code)[0] * np.shape(code)[1])

    FQ = []
    for q in code:
        tmp = int("".join(str(x) for x in q), 2)
        FQ.append(tmp)

    Y = max(FQ) + PERM_PHI

    FC = 0
    for i in range(len(FQ)):
        FC += (FQ[i] * pow(Y, len(FQ) - (i+1)))

    FIT = []
    for i in code:
        for j in i:
            FIT.append(j)
    FIT = int("".join(str(x) for x in FIT), 2)
    return ((FC + FIT) % LD)

def GenerateDCap(code_arr):
    O_LEN = floor(log2(360/DTB_STEP_O)) + 1
    A_LEN = floor(log2(360/DTB_STEP_ALPHA)) + 1
    L_LEN = floor(log2(R/DTB_STEP_L)) + 1

    LD = pow(2, O_LEN + A_LEN + L_LEN + L_LEN)

    result_arr = [0] * int(LD)

    for elem in code_arr:
        result_arr[int(elem)] = 1

    return result_arr

# Feature Decorrelation Algorithm
def FeatDecAlg(code):
    result_code = [0] * LC

    # Step 1 & 2
    for i in range(len(code)):
        if code[i] == 1:
            result_code[i % LC] = 1

    # Step 3 Discrete Fourier Transform
    result_code = list(fft.fft(result_code))

    return result_code

# Enhanced Feature Decorrelation Algorithm
def EnFeatDecAlg(code):

    # Step 1
    P1 = code[:LC]
    P2 = code[LC:]

    # Step 2
    arr = []
    for i in range(NS):
        arr.append(P1[(i * (LC//NS)) : ((i + 1) * (LC//NS))])

    # Step 3
    for i in range(NS):
        phi = 0
        for j in range(LC//NS):
            phi += arr[i][j] * (j+1)
        arr[i] = arr[i][(phi % (LC//NS)):] + arr[i][:(phi % (LC//NS))]

    # Step 4
    P1P = []
    for i in arr:
        for j in i:
            P1P.append(j)

    # Step 5
    result_code = FeatDecAlg(P1P + P2)
    return result_code

# Function for Projecting the feature code
def ProjectVect(vector):

    # Generating Projection Matrix M
    M = []
    random.seed(PROJ_MAT_SEED)
    for i in range(PROJ_MAT_Y_DIM):
        tmp = []
        for j in range(LC):
            tmp.append(random.random())
        M.append(tmp)

    # Vector transformation
    V = np.array(vector)
    M = np.array(M)
    result_vector = list(np.matmul(M, V))

    return result_vector

# Matlab's corr2 function
def corr2(a,b):
    def mean2(x):
        y = np.sum(x) / np.size(x)
        return y
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a*b).sum() / sqrt((a*a).sum() * (b*b).sum())
    return r


# Function for calculating the Final Score between Template and Query
def FinalScore(CT,DT,CQ,DQ):

    SC = []
    for i in range(len(CT)):
        for j in range(len(CQ)):
            A = np.array(CT[i])
            B = np.array(CQ[j])
            tmp = 1 - (linalg.norm(A - B) / (linalg.norm(A) + linalg.norm(B)))
            SC.append(tmp)

    SC = np.array(SC)
    #SC = SC / linalg.norm(SC)
    SC_MAX = np.max(SC)
    print("SC Max: ", SC_MAX)

    SD = corr2(np.array(DT), np.array(DQ))
    print("SD: ", SD)

    final_score = (FINAL_RHO * SC_MAX) + ((1 - FINAL_RHO) * SD)
    print("Final Score: ", final_score)
    return final_score

## Main Program ##

if __name__ == "__main__":

    # Detect and Extract Minutiae of Template image
    print("Template Fingerprint:")
    minutiae_dataframe, minutiae_image = minutiae_extractor.extract(str(sys.argv[1]), False)

    # Extract Minutiae Coordinates and Orientations

    minutiae_points = []
    for _, row in minutiae_dataframe.iterrows():
        minutiae_points.append(Vertex(int(row['x']), int(row['y']), np.rad2deg(row['angle']), int(row['class'])))

    # Generate Delaunay Triangles
    triangulation = delaunay(minutiae_points)

    # Draw Triangles
    """
    for trig in triangulation:
        minutiae_image = cv2.line(minutiae_image, (trig[0][0], trig[0][1]), (trig[1][0], trig[1][1]) , (0,255,0), 1)
        minutiae_image = cv2.line(minutiae_image, (trig[1][0], trig[1][1]), (trig[2][0], trig[2][1]) , (0,255,0), 1)
        minutiae_image = cv2.line(minutiae_image, (trig[0][0], trig[0][1]), (trig[2][0], trig[2][1]) , (0,255,0), 1)
    cv2.imshow('final_output', minutiae_image)
    cv2.waitKey(0)
    """
    # Template Polar
    CT = []
    for i in range(len(minutiae_points)):
        tmp = PCBFeatureCode(minutiae_points, minutiae_points[i])
        tmp = EnFeatDecAlg(tmp)
        tmp = ProjectVect(tmp)
        CT.append(tmp)

    # Template Delaunay
    DT = []
    for i in range(len(triangulation)):
        tmp = DTBFeatureCode(triangulation[i])
        tmp = PermDelFeatCode(tmp)
        DT.append(tmp)
    DT = GenerateDCap(DT)

    # Detect and Extract Minutiae of Query image
    print("Query Fingerprint:")
    minutiae_dataframe, minutiae_image = minutiae_extractor.extract(str(sys.argv[2]), False)

    # Extract Minutiae Coordinates and Orientations

    minutiae_points = []
    for _, row in minutiae_dataframe.iterrows():
        minutiae_points.append(Vertex(int(row['x']), int(row['y']), np.rad2deg(row['angle']), int(row['class'])))

    # Generate Delaunay Triangles
    triangulation = delaunay(minutiae_points)

    # Draw Triangles
    """
    for trig in triangulation:
        minutiae_image = cv2.line(minutiae_image, (trig[0][0], trig[0][1]), (trig[1][0], trig[1][1]) , (0,255,0), 1)
        minutiae_image = cv2.line(minutiae_image, (trig[1][0], trig[1][1]), (trig[2][0], trig[2][1]) , (0,255,0), 1)
        minutiae_image = cv2.line(minutiae_image, (trig[0][0], trig[0][1]), (trig[2][0], trig[2][1]) , (0,255,0), 1)
    cv2.imshow('final_output', minutiae_image)
    cv2.waitKey(0)
    """
    # Query Polar
    CQ = []
    for i in range(len(minutiae_points)):
        tmp = PCBFeatureCode(minutiae_points, minutiae_points[i])
        tmp = EnFeatDecAlg(tmp)
        tmp = ProjectVect(tmp)
        CQ.append(tmp)

    # Query Delaunay
    DQ = []
    for i in range(len(triangulation)):
        tmp = DTBFeatureCode(triangulation[i])
        tmp = PermDelFeatCode(tmp)
        DQ.append(tmp)
    DQ = GenerateDCap(DQ)

    # Calculate Final Score
    FinalScore(CT,DT,CQ,DQ)
