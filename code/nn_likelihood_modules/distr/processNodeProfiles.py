#!/usr/bin/python3

import sys
import math
import re

from sigConstants import *
import histoSigPathManip

#from config3Cl import options

fi = sys.stdin
fo = sys.stdout
fe = sys.stderr

def nodeProfileIterator(filename,
                        options):

    DATA_layerId_POS = 0
    DATA_nodeId_POS = 1
    DATA_nodeContrib_POS = 2

    DATA_EXPECTED_LINE_LEN = 3

    #
    # entry column consistency checks
    #

    #  Modification to accelerate the exterior use
    if options.getStrConfig('opt_process_display', fe) == 'True':
        fo.write('PROCESSING: ' + filename + '\n')
    
    # fo.write('PROCESSING: ' + filename + '\n')
    
    curImgId = histoSigPathManip.getImgId(filename);
    #if opt_test:
    if options.getStrConfig('opt_debug_process_node_profile', fe) == 'True':
        fo.write('IMG ID: ' + str(curImgId) + '\n')

    fd = open(filename, 'r')
    
    line_no = 0
    
    line = fd.readline()
    if line == "":
        fo.write('ERROR: invalid empty data file ' +
                 '\n')
        sys.exit(1)

    line_no = line_no + 1
    line = re.sub('\n$', '', line)
    #words = string.split(line)
    words = line.split()

    #if opt_test:
    if options.getStrConfig('opt_debug_process_node_profile', fe) == 'True':
        fo.write('LINE: ' + line + '\n')
        fo.write('LENGTH: ' + str(len(words)) + '\n')
        #print words
        
    if len(words) < DATA_EXPECTED_LINE_LEN:
        fo.write('ERROR: processNodeProfile: invalid image header line length ' +
                 str(len(words)) +
                 ' at line ' +
                 str(line_no) +
                 ' ' +
                 line +
                 ' (' +
                 str(DATA_EXPECTED_LINE_LEN) +
                 ' expected)' +
                 '\n')
        sys.exit(1)


    if words[DATA_layerId_POS] != 'layerId':
        fo.write('ERROR: processNodeProfile: inconsistent column ' +
                 str(DATA_layerId_POS) +
                 ' ' +
                 words[DATA_layerId_POS] +
                 ' (\"layerId\" expected)'+
                 '\n')
        sys.exit(1)

    if words[DATA_nodeId_POS] != 'nodeId':
        fo.write('ERROR: processNodeProfile: inconsistent column ' +
                 str(DATA_nodeId_POS) +
                 ' ' +
                 words[DATA_nodeId_POS] +
                 ' (\"nodeId\" expected)'+
                 '\n')
        sys.exit(1)

    if words[DATA_nodeContrib_POS] != 'nodeContrib':
        fo.write('ERROR: processNodeProfile: inconsistent column ' +
                 str(DATA_nodeContrib_POS) +
                 ' ' +
                 words[DATA_nodeContrib_POS] +
                 ' (\"nodeContrib\" expected)'+
                 '\n')
        sys.exit(1)
    
    #
    # read table file
    #

    curScore = 0.0
    
    line = fd.readline()
    while line != "":
        line_no = line_no + 1

        #
        # skip empty or comment lines
        #

        if (line == '') or (re.match('^#', line)):
            line = finput.readline()
            continue
        
        line = re.sub('\n$', '', line)
        #words = string.split(line)
        words = line.split()
        
        #if opt_test:
        if options.getStrConfig('opt_debug_process_node_profile', fe) == 'True':
            fo.write('LINE: ' + line + '\n')
            fo.write('LENGTH: ' + str(len(words)) + '\n')
            #print words
        
        if len(words) < DATA_EXPECTED_LINE_LEN:
            fo.write('ERROR: processNodeProfile: invalid image body line length ' +
                     len(words) +
                     ' at line ' +
                     str(line_no) +
                     ' (' +
                     str(DATA_EXPECTED_LINE_LEN) +
                     ' expected)' +
                     '\n')
            sys.exit(1)

        #if opt_test:
        if options.getStrConfig('opt_debug_process_node_profile', fe) == 'True':
            fo.write('WORDS --->\n')
            i = 0
            while i < len(words):
                fo.write('\tWORD ' + str(i) + ': ' + words[i] + '\n')
                i = i + 1
            fo.write('<--- WORDS\n')
            fo.write('\n')


        layerId = words[DATA_layerId_POS]
        nodeId = words[DATA_nodeId_POS]
        
        if options.getStrConfig('opt_debug_process_node_profile', fe) == 'True':
            print('LAYER: ', layerId, '\n')
            print('NODE: ', nodeId, '\n')
            
        try:
            rVal = float(words[DATA_nodeContrib_POS])
        except ValueError:
            fo.write('ERROR: getImgSingleClassLikelihood: invalid value \'' +
                     words[DATA_nodeContrib_POS] + '\' on line ' +
                     str(line_no) + '\n')

        nodeContrib = rVal
            
        yield layerId, nodeId, nodeContrib

        line = fd.readline()
    
    fd.close()

    return


def processNodeProfile(layerId,
                       nodeId,
                       classId,
                       nodeContrib,
                       parNodeStepTable,
                       parNodeProbTable,
                       options):

    nodeKey = (layerId, nodeId, classId)    

    if not nodeKey in parNodeProbTable:
        fe.write('ERROR: processNodeProfile: missing key in signature node prob table ' +
                 str(edgeKey) +
                 '\n')
        sys.exit(1)

    if not nodeKey in parNodeStepTable:
        
        #fo.write('ERROR: processNodeProfile: missing edge ' +
        #         str(edgeKey) +
        #        ' in edge step table ' +
        #         '\n')
        #sys.exit(1)

        if options.getStrConfig('opt_warnings', fe) == 'True':
            fe.write('WARNING: processNodeProfile: missing key in signature node prob table ' +
                     str(nodeKey) +
                     '\n')

        histProb = LOW_SMOOTHED_PROB

        if options.getStrConfig('opt_debug_node_process_profile', fe) == 'True':
            fo.write('DEBUG HISTO NODE: ' +
                     str(nodeKey) +
                     ' CONTRIB: ' +
                     str(nodeContrib) +
                     ' BIN: ' +
                     str(curBinId) +
                     ' PROB: ' +
                     str(histProb) +
                     '\n')
        
        return histProb

    if parNodeStepTable[nodeKey] > 0.0:

        curBinId = int(math.fabs(nodeContrib / parNodeStepTable[nodeKey]))
    else:
        curBinId =  int(math.fabs(nodeContrib))

    #
    #if not nodeKey in parNodeProbTable:
    #    fo.write('ERROR: processNodeProfile: missing node ' +
    #             str(nodeKey) +
    #             ' in node prob table ' +
    #             '\n')
    #    sys.exit(1)
    #
    
    if not curBinId in parNodeProbTable[nodeKey]:

        if options.getStrConfig('opt_warnings', fe) == 'True':
            fo.write('WARNING: processNodeProfile: missing bin ' +
                     str(curBinId) +
                     ' for node ' +
                     str(nodeKey) +
                     ' in node prob table ' +
                     '\n')
            #sys.exit(1)
        
        #histProb = EPSILON
        histProb = LOW_SMOOTHED_PROB
    else:
        histProb = parNodeProbTable[nodeKey][curBinId]

    if options.getStrConfig('opt_debug_node_process_profile', fe) == 'True':
        fo.write('DEBUG HISTO NODE: ' +
                 str(nodeKey) +
                 ' CONTRIB: ' +
                 str(nodeContrib) +
                 ' BIN: ' +
                 str(curBinId) +
                 ' PROB: ' +
                 str(histProb) +
                 '\n')
        
    return histProb

