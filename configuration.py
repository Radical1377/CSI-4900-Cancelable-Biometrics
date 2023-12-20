R = 300 # Max Radial Distance

SHOW_PIC = False

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
NS = 20000 # Must divide LC

# Projection Matrix Params
PROJ_MAT_SEED = 1337 # Must be 32-bit integer
PROJ_MAT_Y_DIM = 50 # Must be between 1-L*H*S

# Delaunay Code Permutation Params
PERM_PHI = 14

# Final Score Params
FINAL_RHO = 0.7

# are we using the 1v1 protocol
IS_1V1 = True

# Output NPY file name
NPY_FILE_NAME = "SPEEDTEST.npy"

# Are we using the enhanced version of the Decorrelation algorithm
IS_DBA_ENHANCED = False

DEL_TFT_LEN = 20000
