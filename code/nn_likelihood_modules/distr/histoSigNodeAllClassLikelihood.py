#!/usr/bin/python3

import sys
import string
import re
import getopt
import math
import histoSigPathManip
#from scipy.stats import norm

from config3Cl import options
from histoSigNodeRead import readHistoSigNodeTable

from sigConstants import *

from histoSigPathManip import getImgId

from processEntries import entryIterator
from processNodeProfiles import nodeProfileIterator, processNodeProfile
from readClassList import readClassList

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

        
def getImgNodeAllClassLikelihood(filename,
                                 classList,
                                 parNodeStepTable,
                                 parNodeProbTable,
                                 options):

    curScoreArr = {}

    #for bLayer, eLayer, bNode, eNode, classId, edgeProfile in edgeProfileIterator(filename,
    #for bLayer, eLayer, bNode, eNode, edgeProfile in edgeProfileIterator(filename,
    for layerId, nodeId, nodeProfile in nodeProfileIterator(filename,
                                                            options):


        #
        # all class loop
        #

        for classId in classList:

            histProb = processNodeProfile(layerId,
                                          nodeId,
                                          classId,
                                          nodeProfile,
                                          parNodeStepTable,
                                          parNodeProbTable,
                                          options)
        
            if not classId in curScoreArr:
                curScoreArr[classId] = 0.0
            
            curScoreArr[classId] = curScoreArr[classId] - math.log(histProb)

    return(curScoreArr)


def computeNodeAllClassLikelihood(imgDir,
                                  imgListFileName,
                                  classList,
                                  parNodeStepTable,
                                  parNodeProbTable,                             
                                  options):

    for path in entryIterator(imgDir, imgListFileName, options):

        imgId = getImgId(path)

        imgScoreArr = getImgNodeAllClassLikelihood(path,
                                                   classList,
                                                   parNodeStepTable,
                                                   parNodeProbTable,
                                                   options)

        minDist = 9999.9999
        minClass = -9999
    
        for clId in sorted(imgScoreArr):
            fo.write('CLASS: ' +
                     str(clId) +
                     #' SCORE: ' +
                     ' ' +
                     str(round(imgScoreArr[clId], F_PREC)) +
                     '\n')
        
            fo.write('TAB: ' +
                     'dist' +
                     ' ' +
                     imgId +
                     ' ' +
                     str(clId) +
                     ' ' +
                     str(round(imgScoreArr[clId], F_PREC)) +
                     '\n')
        
            if (imgScoreArr[clId] < minDist):
                minDist = imgScoreArr[clId]
                minClass = clId

        fo.write('BEST: ' +
                 minClass +
                 ' ' +
                 str(round(minDist, F_PREC)) +
                 '\n')


        fo.write('TAB: ' +
                 'bestDist' +
                 ' ' +
                 imgId +
                 ' ' +
                 minClass +
                 ' ' +
                 str(round(minDist, F_PREC)) +
                 '\n')

    return


#
#if len(sys.argv) != 4:
#    fe.write('ERROR: invalid number of arguments ' +
#             str(len(sys.argv) - 1) +
#             ' (3 expected)\n')
#    sys.exit(1)     
#
#
#sigHistoFileName = sys.argv[1]
#imgDir = sys.argv[2]
#imgListFileName = sys.argv[3]
#

#print 'ARGS: ', args


if len(args) != 4:
    fe.write('ERROR: invalid number of arguments ' +
             str(len(args)) +
             ' (4 expected)\n')
    sys.exit(1)     


sigHistoFileName = args[0]
imgDir = args[1]
imgListFileName = args[2]
classListFileName = args[3]


options.readFileNameConfig('options.dat', fe, fo)
if (options.getStrConfig("printOptions", fe) == 'True'):
        options.printConfig(fe, fo)

nodeStepTable = {}
nodeProbTable = {}

#classList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
classList = readClassList(classListFileName, options)

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

computeNodeAllClassLikelihood(imgDir,
                              imgListFileName,
                              classList,
                              nodeStepTable,
                              nodeProbTable,
                              options)

sys.exit(0)
