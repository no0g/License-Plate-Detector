#!/bin/env python3

# Importing Libraries
import argparse
import cv2 # OpenCV for Computer Vision
import numpy as np # For processing numbers
import imutils # For image processing
from imutils import paths
import easyocr # OCR library
import os.path
import json # to parse test label



# preprocess images by converting to gray
def preProcessing(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Noise Reduction and Edge Detection
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) #Edge detection
    # plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)) # For debug
    return edged,gray

def findLicensePlate(edged):    
    # Find license plate
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # Loop through the image to find the countour
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break

    return location,approx

def maskAndCrop(location,img,gray):
    # Define mask and place to crop
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    # Cropping image
    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    return cropped_image

def getLicenseText(cropped_image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image,paragraph="False")
    if result: 
        if type(result[0][1]) != type("lol"):
            return None
        return result[0][1].replace("]","").replace("[","")
    else:
        return None

def detectLicensePlate(filename,debug=False):
    img = cv2.imread(filename)
    edged, gray = preProcessing(img)
    location,approx = findLicensePlate(edged)
    if location is not None:
        cropped_image = maskAndCrop(location,img,gray)
        result = getLicenseText(cropped_image)
        #print(f"[+] Detection Result: {result}, Source File: {filename}")
        if debug:
            cv2.imshow("Source",img)
            cv2.imshow("Edged",edged)
            cv2.imshow("Cropped",cropped_image)
            cv2.imshow("Result",showResult(filename,result,approx))
            cv2.waitKey(0)
        return result,approx
    else:
        return None,approx 

def showResult(filename,result,approx):
    img = cv2.imread(filename)
    text = result
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img, text=text, org=(200,200), fontFace=font, fontScale=1,
                      color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
    cv2.imshow("Result",cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple License Plate Detection using OpenCV and EasyOCR ')
    parser.add_argument("-i", "--input",
	help="path to an image")
    parser.add_argument("-s", "--show", action="store_true",
	help="show final visualizations")
    parser.add_argument("-d", "--debug", action="store_true",
	help="show all image processing steps")
    parser.add_argument("-v","--validate",action="store_true",
    help="validate result with test images")
    args = vars(parser.parse_args())

    show = args["show"]
    debug = args["debug"]

    # Colors
    cyan  = "\033[0;96m"
    green   = "\033[0;92m"
    white   = "\033[0;97m"
    red   = "\033[0;91m"
    blue  = "\033[0;94m"
    yellow  = "\033[0;33m"
    magenta = "\033[0;35m"
    Color_Off='\033[0m'

    if args["validate"]:
        testImages = sorted(list(paths.list_images("./test-images/")))
        with open('./test-label/testdata.json') as f:
            testLabels = json.loads(f.read())
        correct = 0
        total = 0
        for image in testImages:
            license_text,approx = detectLicensePlate(image)
            if license_text is not None:
                print(green + "[+] Detection Result: {}, Source File: {}".format(license_text.upper(),image)+Color_Off)
                if testLabels[os.path.basename(image)] ==  license_text.upper():
                    correct+=1
                    print("\t"+green+"[++] Detected Correctly"+Color_Off)
                else:
                    print("\t"+red+"[--] Detected but wrong value"+", Correct Value: "+testLabels[os.path.basename(image)]+Color_Off)
            else:
                print(red+"[-] Could not find license plate!"+" Source: "+image+Color_Off)
                print("\t"+red+"[--] Failed to detect"+", Correct Value: "+testLabels[os.path.basename(image)]+Color_Off)
            total+=1
        print("\n==========Validation Result===========\n")
        print("Correctly Detected: "+str(correct)+"\nTotal Image Tested: "+str(total)+"\nAccuracy: "+str(correct/total*100/100))
        print("\n======================================\n")

    elif args["input"]:
        filename = args["input"]
        if not os.path.isfile(filename):
            print(red+"[-] Could not find file!"+Color_Off)
            exit(0)
        if show:
            license_text,approx = detectLicensePlate(filename)
            if license_text is not None:
                print(green + "[+] Detection Result: {}, Source File: {}".format(license_text,filename)+Color_Off)
                showResult(filename,license_text,approx)
            else:
                print(red+"[-] Could not find license plate!"+Color_Off)
        else:
            license_text,approx = detectLicensePlate(filename,debug)
            if license_text is not None:
                print(green + "[+] Detection Result: {}, Source File: {}".format(license_text,filename)+Color_Off)
            else:
                print(red+"[-] Could not find license plate!"+Color_Off)



        

