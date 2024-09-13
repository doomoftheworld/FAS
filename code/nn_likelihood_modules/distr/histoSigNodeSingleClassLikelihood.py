#!/usr/bin/python3

import sys
import string
import re
import getopt
import math
import os
#from scipy.stats import norm

from config3Cl import options
from histoSigNodeRead import readHistoSigNodeTable
from sigConstants import *

from histoSigPathManip import getImgId

from processEntries import entryIterator
from processNodeProfiles import nodeProfileIterator, processNodeProfile

from multiprocessing.pool import Pool


fi = sys.stdin
fo = sys.stdout
fe = sys.stderr

def usage():
    sys.stdout.write('\n')
    sys.stdout.write('Usage: ' + sys.argv[0] + ' [-h]\n\n')
    sys.stdout.write('Options:\n')
    sys.stdout.write('  -h --help : this help\n')
    sys.stdout.write('     --in_separator : input separator (blank, tab, comma, semicolon)\n')
    sys.stdout.write('     --out_separator : output separator (blank, tab, comma, semicolon)\n')
    #sys.stdout.write('  -z --zero : columns start from zero\n\n\
    sys.stdout.write('Prints the values and positions of fields in tabular form from an input file\n')
    sys.stdout.write('\n')

try:
    #opts, args = getopt.getopt(sys.argv[1:], "hz", ["help", "in_separator=",
    #    "out_separator=", "zero"])
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help", "in_separator=",
                                           "out_separator="])

except getopt.error:
    usage()
    sys.exit(1)

for o, v in opts:
    if o in ("-h", "--help"):
        usage()
        sys.exit(0)
    elif o == "--in_separator":
        if v == 'blank':
            opt_i_sep = 'true'
            i_sep_ch = ' '
        elif v == 'tab':
            opt_i_sep = 'true'
            i_sep_ch = '\t'
        elif v == 'comma':
            opt_i_sep = 'true'
            i_sep_ch = ','
        elif v == 'semicolon':
            opt_i_sep = 'true'
            i_sep_ch = ';'
        elif v == 'ampersand':
            i_sep_ch = '&'
            opt_i_sep = 'true'
        else:
            fe.write('ERROR: invalid input separator ' + v + '\n')
            sys.exit(1)
    elif o == "--out_separator":
        if v == 'blank':
            opt_o_sep = 'true'
            o_sep_ch = ' '
        elif v == 'tab':
            opt_o_sep = 'true'
            o_sep_ch = '\t'
        elif v == 'comma':
            opt_o_sep = 'true'
            o_sep_ch = ','
        elif v == 'semicolon':
            opt_o_sep = 'true'
            o_sep_ch = ';'
        elif v == 'ampersand':
            o_sep_ch = '&'
            opt_o_sep = 'true'
        else:
            fe.write('ERROR: invalid output separator ' + v + '\n')
            sys.exit(1)
    #
    #elif o in ("-z", "--zero"):
    #    init_col = 0
    #
    elif o == "--header":
        if (v == 'on') or (v == 'off'):
            hop = v;
        else:
            fe.write('ERROR: invalid header option ' + v + ' (on / off expected)\n')
            sys.exit(1)     
    else:
        fe.write('ERROR: cannot process option ' + o + '\n')
        sys.exit(1)

        
def getImgNodeSingleClassLikelihood(filename,
                                    parNodeStepTable,
                                    parNodeProbTable,
                                    parClass,
                                    options):

    curScore = 0.0

    #fo.write('DEBUG: PAR_CLASS: ' + parClass + '\n')
    
    #for bLayer, eLayer, bNode, eNode, classId, edgeProfile in edgeProfileIterator(filename,
    for layerId, nodeId, nodeProfile in nodeProfileIterator(filename,
                                                            options):

#        
#        if parClass != classId:
#            line = fd.readline()
#            continue
#

        histProb = processNodeProfile(layerId,
                                      nodeId,
                                      parClass,
                                      nodeProfile,
                                      parNodeStepTable,
                                      parNodeProbTable,
                                      options)
        
        curScore = curScore - math.log(histProb)

    return(curScore)


def computeNodeSingleClassLikelihood(imgDir,
                                     imgListFileName,
                                     parNodeStepTable,
                                     parNodeProbTable,                             
                                     targetClass,
                                     options):

    for path in entryIterator(imgDir, imgListFileName, options):

        imgId = getImgId(path)

        imgScore = getImgNodeSingleClassLikelihood(path,
                                                   parNodeStepTable,
                                                   parNodeProbTable,
                                                   targetClass,
                                                   options)
        
        #if opt_test:
        if options.getStrConfig('opt_test', fe) == 'True':
            fo.write('SCORE: ' + str(round(imgScore, F_PREC)) + '\n')

        fo.write('TAB: ' +
                 'dist' +
                 ' ' +
                 imgId +
                 ' ' +
                 targetClass +
                 ' ' +
                 str(round(imgScore, F_PREC)) +
                 '\n')
    return
    
#print 'ARGS: ', args

if len(args) != 4:
    fe.write('ERROR: invalid number of arguments ' +
             str(len(args)) +
             ' (4 expected)\n')
    sys.exit(1)     

sigHistoFileName = args[0]
imgDir = args[1]
imgListFileName = args[2]             
targetClass = args[3]

options.readFileNameConfig('options.dat', fe, fo)
if (options.getStrConfig("printOptions", fe) == 'True'):
        options.printConfig(fe, fo)

if options.getStrConfig('opt_test', fe) == 'True':
    fo.write('ARGS: ' +
             'sigHistoFileName: ' + sigHistoFileName + ', ' +
             'imgDir: ' + imgDir + ', ' +
             'imgListFileName: ' + imgListFileName + ', ' +
             'targetClass: ' + targetClass +
             '\n')

nodeStepTable = {}
nodeProbTable = {}

#
# read sig table
#

fd = open(sys.argv[1], 'r')
fd = open(sigHistoFileName, 'r')

readHistoSigNodeTable(fd,
                      nodeStepTable,
                      nodeProbTable,
                      options)

#
# read / process entries
#

computeNodeSingleClassLikelihood(imgDir,
                                 imgListFileName,
                                 nodeStepTable,
                                 nodeProbTable,
                                 targetClass,
                                 options)

sys.exit(0)
