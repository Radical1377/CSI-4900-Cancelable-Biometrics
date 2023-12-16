import os, cv2, sys, copy, random, json, time
import numpy as np
import scipy.stats as stats
from math import *
from numpy import fft, linalg
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from numba import njit, jit, cuda
from delaunay_triangulation.triangulate import delaunay
from delaunay_triangulation.typing import Vertex

R = 300 # Max Radial Distance

SHOW_PIC = True

# Minutiae Score threshold
MIN_SCORE_THRESH = 0.0

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
PROJ_MAT_Y_DIM = 300 # Must be between 1-LC

# Delaunay Code Permutation Params
PERM_PHI = 1

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
    edge_len = [[{0,1}, 0], [{1,2}, 0], [{0,2}, 0]]
    #[0,1]
    edge_len[0][1] = sqrt(pow(triangle[0][0]-triangle[1][0], 2) + pow(triangle[0][1]-triangle[1][1], 2))
    #[1,2]
    edge_len[1][1] = sqrt(pow(triangle[1][0]-triangle[2][0], 2) + pow(triangle[1][1]-triangle[2][1], 2))
    #[0,2]
    edge_len[2][1] = sqrt(pow(triangle[0][0]-triangle[2][0], 2) + pow(triangle[0][1]-triangle[2][1], 2))

    print("EDGE LEN: ", edge_len)

    # Calculating all angles and finding the largest one
    vert_ang = [[0,0], [1,0], [2,0]]
    # 0
    vert_ang[0][1] = degrees(acos((pow(edge_len[0][1],2) + pow(edge_len[2][1],2) - pow(edge_len[1][1],2)) / (2 * edge_len[0][1] * edge_len[2][1])))
    # 1
    vert_ang[1][1] = degrees(acos((pow(edge_len[0][1],2) + pow(edge_len[1][1],2) - pow(edge_len[2][1],2)) / (2 * edge_len[0][1] * edge_len[1][1])))
    # 2
    vert_ang[2][1] = degrees(acos((pow(edge_len[1][1],2) + pow(edge_len[2][1],2) - pow(edge_len[0][1],2)) / (2 * edge_len[1][1] * edge_len[2][1])))

    print("VERT ANG: ", vert_ang)

    tmp = 0
    for ang in vert_ang:
        if ang[1] >= tmp:
            max_ang = ang
            tmp = ang[1]

    A = max_ang[1]
    print("MAX ANG: ", max_ang)

    # Setting up L1 and L2
    selected_edges = []
    for edge in edge_len:
        if max_ang[0] in edge[0]:
            selected_edges.append(edge)
    print("SEL EDGES: ", selected_edges)

    if selected_edges[0][1] >= selected_edges[1][1]:
        big_len = selected_edges[0]
        sma_len = selected_edges[1]
    else:
        big_len = selected_edges[1]
        sma_len = selected_edges[0]

    L1 = big_len[1]
    L2 = sma_len[1]

    print("BIG LEN: ", big_len)
    print("SMA LEN: ", sma_len)

    """
    # Calculating O_CODE
    O = abs(triangle[1][2] - triangle[2][2])
    O = bin(floor(O / DTB_STEP_O))[2:]
    O = ('0' * (O_LEN - len(O) - 1)) + O
    O_CODE = []
    for i in O:
        O_CODE.append(int(i))
    """

    # Calculating A_CODE
    A = bin(floor(A / DTB_STEP_ALPHA))[2:]
    A = ('0' * (A_LEN - len(A) - 1)) + A
    A_CODE = []
    for i in A:
        A_CODE.append(int(i))

    # Calculating L1_CODE
    L1 = bin(floor(L1 / DTB_STEP_L))[2:]
    L1 = ('0' * (L_LEN - len(L1) - 1)) + L1
    L1_CODE = []
    for i in L1:
        L1_CODE.append(int(i))

    # Calculating L2_CODE
    L2 = bin(floor(L2 / DTB_STEP_L))[2:]
    L2 = ('0' * (L_LEN - len(L2) - 1)) + L2
    L2_CODE = []
    for i in L2:
        L2_CODE.append(int(i))

    # Calculating T_CODE
    big_ind = big_len[0]
    big_ind.remove(max_ang[0])
    big_ind = big_ind.pop()

    sma_ind = sma_len[0]
    sma_ind.remove(max_ang[0])
    sma_ind = sma_ind.pop()

    T_CODE = [int(triangle[big_ind][3] == 0),
              int(triangle[max_ang[0]][3] == 0),
              int(triangle[sma_ind][3] == 0)]

    # Concatenate all CODE binary arrays to generate the feature code
    feature_code = [A_CODE, L1_CODE, L2_CODE, T_CODE]
    return feature_code

# Function for permuting the Delaunay Feature code
def PermDelFeatCode(code):
    LD = pow(2, 20)

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
    return (FC + FIT)

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
@jit(target_backend='cuda')
def ProjectVect(vector):

    # Generating Projection Matrix M

    np.random.seed(PROJ_MAT_SEED)
    M = np.random.rand(PROJ_MAT_Y_DIM, LC)

    # Vector transformation
    V = np.array(vector)
    result_vector = list(np.dot(M, V))

    return result_vector

# Matlab's corr2 function


def corr2(a,b):

    a = a - (np.sum(a) / pow(2,20))
    b = b - (np.sum(b) / pow(2,20))
    r = (a*b).sum() / sqrt((a*a).sum() * (b*b).sum())
    return r

# Function for calculating raw SC_MAX of a fingerprint pair
def calcSCMAX(CT, CQ):
    SC = []
    for i in range(len(CT)):
        for j in range(len(CQ)):
            A = np.array(CT[i])
            B = np.array(CQ[j])
            tmp = 1 - (linalg.norm(A - B) / (linalg.norm(A) + linalg.norm(B)))
            SC.append(tmp)

    SC = np.array(SC)
    SC_MAX = np.max(SC)
    return SC_MAX

# Function for calculating raw SD of a fingerprint pair
def calcSD(DT, DQ):
    SD = corr2(np.array(DT), np.array(DQ))
    return SD

# Function for calculating the Final Score between normalized SC_MAX and SD
def FinalScore(SC_MAX, SD):

    final_score = (FINAL_RHO * SC_MAX) + ((1 - FINAL_RHO) * SD)
    return final_score

## Main Program ##

if __name__ == "__main__":

    # if directory is given as argument
    if os.path.isdir(sys.argv[1]):

        import minutiae_extractor

        files = os.listdir(sys.argv[1])
        json_output = {
            "DIRECTORY": sys.argv[1],
            "FILES": files
        }

        for file in files:

            # Detect and Extract Minutiae of Template image
            print("Processing: ", file)
            minutiae_dataframe, minutiae_image = minutiae_extractor.extract(str(sys.argv[1]) + file, SHOW_PIC)

            # Extract Minutiae Coordinates and Orientations

            minutiae_points = []
            for _, row in minutiae_dataframe.iterrows():
                if float(row['score']) > MIN_SCORE_THRESH:
                    minutiae_points.append(Vertex(int(row['x']), int(row['y']), np.rad2deg(row['angle']), int(row['class'])))

            # Generate Delaunay Triangles
            triangulation = delaunay(minutiae_points)

            # Draw Triangles
            if SHOW_PIC:
                for trig in triangulation:
                    minutiae_image = cv2.line(minutiae_image, (trig[0][0], trig[0][1]), (trig[1][0], trig[1][1]) , (0,255,0), 2)
                    minutiae_image = cv2.line(minutiae_image, (trig[1][0], trig[1][1]), (trig[2][0], trig[2][1]) , (0,255,0), 2)
                    minutiae_image = cv2.line(minutiae_image, (trig[0][0], trig[0][1]), (trig[2][0], trig[2][1]) , (0,255,0), 2)
                cv2.imshow('final_output', minutiae_image)
                cv2.waitKey(0)

            # Polar Coordinate Template
            CT = []
            for i in range(len(minutiae_points)):

                start = time.time()
                tmp = PCBFeatureCode(minutiae_points, minutiae_points[i])
                end = time.time()
                print("PCBFeatureCode Execution time:", end-start)

                start = time.time()
                tmp = EnFeatDecAlg(tmp)
                end = time.time()
                print("EnFeatDecAlg Execution time:", end-start)

                start = time.time()
                tmp = ProjectVect(tmp)
                end = time.time()
                print("ProjectVect Execution time:", end-start)

                CT.append(tmp)

            # Delaunay Triangle Template
            DT = []
            for i in range(len(triangulation)):

                start = time.time()
                tmp = DTBFeatureCode(triangulation[i])
                end = time.time()
                print("DTBFeatureCode Execution time:", end-start)

                start = time.time()
                tmp = PermDelFeatCode(tmp)
                end = time.time()
                print("PermDelFeatCode Execution time:", end-start)

                DT.append(tmp)

            start = time.time()
            DT = GenerateDCap(DT)
            end = time.time()
            print("GenerateDCap Execution time:", end-start)

            json_output[file] = {"CT": CT, "DT": DT}

        with open("test.npy", "wb") as outfile:
            np.save(outfile, np.array([json_output]))

    # if JSON file is given as argument
    elif os.path.isfile(sys.argv[1]) and str(sys.argv[1])[-4:] == '.npy':

        # Load JSON
        print("Loading NPY...")
        fingerprints = np.load(sys.argv[1], allow_pickle=True)[0]

        files = fingerprints["FILES"]
        """
        files = []
        for file in fingerprints["FILES"]:
            if ("_1" in file) or ("_2" in file):
                files.append(file)
        """

        """
            Processed Data row format: [file1, file2, raw/norm sc_max, raw/norm sd, final_score]
        """
        processed_data = []

        # Calculating raw SC_MAXs and SDs
        for i in range(len(files) - 1):
            for j in range(i+1, len(files)):
                data = [files[i], files[j]]
                print(files[i], ' vs ', files[j])

                start = time.time()
                SC_MAX = calcSCMAX(fingerprints[files[i]]["CT"], fingerprints[files[j]]["CT"])
                end = time.time()
                print("calcSCMAX Execution time:", end-start)

                data.append(SC_MAX)

                start = time.time()
                SD = calcSD(fingerprints[files[i]]["DT"], fingerprints[files[j]]["DT"])
                end = time.time()
                print("calcSD Execution time:", end-start)

                data.append(SD)
                processed_data.append(data)

        # Normalizing SC_MAXs and SDs
        SC_MAXs = list(zip(*processed_data))[2]
        SDs = list(zip(*processed_data))[3]

        for i in range(len(processed_data)):
            processed_data[i][2] = (processed_data[i][2] - min(SC_MAXs)) / (max(SC_MAXs) - min(SC_MAXs))
            processed_data[i][3] = (processed_data[i][3] - min(SDs)) / (max(SDs) - min(SDs))

        # Adding final scores
        for row in processed_data:
            row.append(FinalScore(row[2], row[3]))

        # Calculating FRR and FAR
        threshold_step = 0.001
        FAR = []
        FRR = []

        for threshold in np.arange(0.0, 1.0, threshold_step):
            print(threshold)
            accepted_imposter = 0
            total_imposter = 0
            rejected_genuine = 0
            total_genuine = 0
            for row in processed_data:
                # FAR Calculation
                if row[0].split('_')[0] != row[1].split('_')[0]:
                    total_imposter += 1
                    if row[4] > threshold:
                        accepted_imposter += 1

                # FRR Calculation
                if row[0].split('_')[0] == row[1].split('_')[0]:
                    total_genuine += 1
                    if row[4] < threshold:
                        rejected_genuine += 1

            FAR.append(accepted_imposter / total_imposter)
            FRR.append(rejected_genuine / total_genuine)

        # Plotting FAR and FRR based on threshold
        plt.plot(list(np.arange(0.0, 1.0, threshold_step)),FRR,
                list(np.arange(0.0, 1.0, threshold_step)), FAR)
        plt.show()

        # ROC Curve
        plt.plot(FAR, FRR)
        plt.show()
