import cv2, sys, os, time
from fingerflow.extractor import Extractor


COARSENET_PATH = os.getcwd() + '\\models\\CoarseNet.h5'
FINENET_PATH = os.getcwd() + '\\models\\FineNet.h5'
CLASSIFYNET_PATH = os.getcwd() + '\\models\\ClassifyNet_6_classes.h5'
CORENET_PATH = os.getcwd() + '\\models\\CoreNet.weights'

extractor = Extractor(COARSENET_PATH, FINENET_PATH, CLASSIFYNET_PATH, CORENET_PATH)

# Returns pandas DataFrame of all minutiaes
def extract(file_name, show=True):

    image = cv2.imread(file_name)
    extracted_minutiae = extractor.extract_minutiae(image)

    if show:
        for _, row in extracted_minutiae['minutiae'].iterrows():
            image = cv2.circle(image, (int(row['x']), int(row['y'])), 5, (255,0,0), 2)
        print(extracted_minutiae['minutiae'])
        cv2.imshow('minutiae', image)
        cv2.waitKey(1)

    return (extracted_minutiae['minutiae'], image)

if __name__ == "__main__":
    start = time.time()
    extract(str(sys.argv[1]), show = True)
    end = time.time()
    print("Execution time:", end-start)
