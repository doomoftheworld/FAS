#!/usr/bin/python3

import os
import sys

def getImgId(parPath):

    pathArr = os.path.split(parPath);
    #sys.stdout.write('PAR PATH: ' + str(pathArr) + '\n')

    #imgName = pathArr[1]
    imgName = pathArr[len(pathArr) - 1]
    #sys.stdout.write('IMG NAME: ' + str(imgName) + '\n')

    imgArr = imgName.split('_')
    #sys.stdout.write('IMG ARR: ' + str(imgArr) + '\n')

    imgId = imgArr[1]
    #sys.stdout.write('RET IMG ID: ' + str(imgId) + '\n')
    
    return imgId
