#!/usr/bin/python3

import sys
import math
import re
from sigConstants import *

#from config3Cl import options

fi = sys.stdin
fo = sys.stdout
fe = sys.stderr

def readHistoSigNodeTable(fd,
                          parNodeStepTable,
                          parNodeProbTable,
                          options):

    nodeHistTable = {}
    nodeFreqTable = {}
    sigmaLbTable = {}
    sigmaUbTable = {}

    parNodeStepTable.clear()
    parNodeProbTable.clear()

    SIG_layerId_POS = 0
    SIG_nodeId_POS = 1
    SIG_classId_POS = 2
    SIG_binId_POS = 3
    SIG_sigmaInterval_lb_POS = 4
    SIG_sigmaInterval_ub_POS = 5
    SIG_sigmaFreq_POS = 6

    SIG_EXPECTED_LINE_LEN = 7


    NODE_KEY_layerId_POS = 0
    NODE_KEY_nodeId_POS = 1
    NODE_KEY_classId_POS = 2

    

    #
    # signature column consistency checks
    #
    
    line_no = 0

    line = fd.readline()
    if line == "":
        fo.write('ERROR: invalid empty signature file ' +
                 '\n')
        sys.exit(1)

    line_no = line_no + 1
    line = re.sub('\n$', '', line)
    #words = string.split(line)
    words = line.split()

    #if opt_test:
    if options.getStrConfig('opt_debug_read_histo_node_table', fe) == 'True':
        fo.write('LINE: ' + line + '\n')
        fo.write('LENGTH: ' + str(len(words)) + '\n')
        #print words
        
    if len(words) != SIG_EXPECTED_LINE_LEN:
        fo.write('ERROR:readHistoSigNodeTable: invalid signature header line length ' +
                 str(len(words)) +
                 ' at line ' +
                 str(line_no) +
                 ' ' +
                 line +
                 ' (' +
                 str(SIG_EXPECTED_LINE_LEN) +
                 ' expected)' +
                 '\n')
        sys.exit(1)


    if words[SIG_layerId_POS] != 'layerId':
        fo.write('ERROR:readHistoSigNodeTable: inconsistent column ' +
                 str(SIG_layerId_POS) +
                 ' ' +
                 words[SIG_layerId_POS] +
                 ' (\"layerId\" expected)'+
                 '\n')
        sys.exit(1)

    if words[SIG_nodeId_POS] != 'nodeId':
        fo.write('ERROR:readHistoSigNodeTable: inconsistent column ' +
                 str(SIG_nodeId_POS) +
                 ' ' +
                 words[SIG_nodeId_POS] +
                 ' (\"nodeId\" expected)'+
                 '\n')
        sys.exit(1)

    if words[SIG_classId_POS] != 'classId':
        fo.write('ERROR:readHistoSigNodeTable: inconsistent column ' +
                 str(SIG_classId_POS) +
                 ' ' +
                 words[SIG_classId_POS] +
                 ' (\"classId\" expected)'+
                 '\n')
        sys.exit(1)

    if words[SIG_binId_POS] != 'binId':
        fo.write('ERROR:readHistoSigNodeTable: inconsistent column ' +
                 str(SIG_binId_POS) +
                 ' ' +
                 words[SIG_binId_POS] +
                 ' (\"binId\" expected)'+
                 '\n')
        sys.exit(1)

    if words[SIG_sigmaInterval_lb_POS] != 'sigmaInterval_lb':
        fo.write('ERROR:readHistoSigNodeTable: inconsistent column ' +
                 str(SIG_sigmaInterval_lb_POS) +
                 ' ' +
                 words[SIG__sigmaInterval_lbPOS] +
                 ' (\"sigmaInterval_lb\" expected)'+
                 '\n')
        sys.exit(1)

    if words[SIG_sigmaInterval_ub_POS] != 'sigmaInterval_ub':
        fo.write('ERROR:readHistoSigNodeTable: inconsistent column ' +
                 str(SIG_sigmaInterval_ub_POS) +
                 ' ' +
                 words[SIG_sigmaInterval_ub_POS] +
                 ' (\"sigmaInterval_ub\" expected)'+
                 '\n')
        sys.exit(1)

    if words[SIG_sigmaFreq_POS] != 'sigmaFreq':
        fo.write('ERROR:readHistoSigNodeTable: inconsistent column ' +
                 str(SIG__sigmaFreqPOS) +
                 ' ' +
                 words[SIG__sigmaFreqPOS] +
                 ' (\"sigmaFreq\" expected)'+
                 '\n')
        sys.exit(1)


    #
    # read table file
    #
    
    line = fd.readline()
    while line != "":
        line_no = line_no + 1

        #
        # skip empty or comment lines
        #

        if (line == '') or (re.match('^#', line)):
            line = fd.readline()
            continue

        line = re.sub('\n$', '', line)
        words = line.split()

        #if opt_test:
        if options.getStrConfig('opt_debug_read_histo_node_table', fe) == 'True':
            fo.write('LINE: ' + line + '\n')
            fo.write('LENGTH: ' + str(len(words)) + '\n')
            #print words
        
        if len(words) != SIG_EXPECTED_LINE_LEN:
            fo.write('ERROR:readHistoSigNodeTable: invalid signature body line length ' +
                     str(len(words)) +
                     ' at line ' +
                     str(line_no) +
                     ' ' +
                     line +
                     ' (' +
                     str(SIG_EXPECTED_LINE_LEN) +
                     ' expected)' +
                     '\n')
            sys.exit(1)

        layerId = words[SIG_layerId_POS]
        nodeId = words[SIG_nodeId_POS]
        classId = words[SIG_classId_POS]

        nodeKey = (layerId, nodeId, classId)

        try:
            iVal = int(words[SIG_binId_POS])
        except ValueError:
            fo.write('ERROR:readHistoSigNodeTable: invalid binId integer value \'' +
                     words[SIG_binId_POS] + '\' on line ' +
                     str(line_no) + '\n')
            sys.exit(1)
            
        binId = iVal

        try:
            rVal = float(words[SIG_sigmaInterval_lb_POS])
        except ValueError:
            fo.write('ERROR:readHistoSigNodeTable: invalid sigma lb float value \'' +
                     words[SIG_sigmaInterval_lb_POS] + '\' on line ' +
                     str(line_no) + '\n')
            sys.exit(1)
            
        sigmaLb = rVal
        
        if  not nodeKey in sigmaLbTable:
            sigmaLbTable[nodeKey] = {}

        if binId in sigmaLbTable[nodeKey]:
                fo.write('ERROR:readHistoSigNodeTable: invalid duplicate binId ' +
                         str(binId) +
                         ' on line ' +
                         str(line_no) +
                         ' in sigma lb table ' +
                         '\n')
                sys.exit(1)
                
        sigmaLbTable[nodeKey][binId] = sigmaLb

        try:
            rVal = float(words[SIG_sigmaInterval_ub_POS])
        except ValueError:
            fo.write('ERROR:readHistoSigNodeTable: invalid sigma ub float value \'' +
                     words[SIG_sigmaInterval_ub_POS] + '\' on line ' +
                     str(line_no) + '\n')
            sys.exit(1)
            
        sigmaUb = rVal
        
        if  not nodeKey in sigmaUbTable:
            sigmaUbTable[nodeKey] = {}

        if binId in sigmaUbTable[nodeKey]:
                fo.write('ERROR:readHistoSigNodeTable: invalid duplicate binId ' +
                         str(binId) +
                         ' on line ' +
                         str(line_no) +
                         ' in sigma ub table ' +
                         '\n')
                sys.exit(1)
                
        sigmaUbTable[nodeKey][binId] = sigmaUb


        nodeStep = math.fabs(sigmaUb - sigmaLb)
        
        if nodeStep < EPSILON:
            #if opt_warnings:
            if options.getStrConfig('opt_warnings', fe) == 'True':
                fo.write('WARNING:readHistoSigNodeTable: node ' +
                         str(nodeKey) +
                         ' binId ' +
                         str(binId) +
                         ' has a null variance' +
                         '\n')
            nodeStep = 0.0

        if not nodeKey in parNodeStepTable:
            parNodeStepTable[nodeKey] = nodeStep
        else:
            if not math.fabs(parNodeStepTable[nodeKey] - nodeStep) < EPSILON:
                fo.write('ERROR:readHistoSigNodeTable: inconsistent steps for same node ' +
                         str(nodeKey) +
                         ' cur step ' +
                         str(nodeStep) +
                         ' and prev step ' +
                         str(parNodeStepTable[nodeKey]) +
                         '\n')
                sys.exit(1)

        try:
            iVal = int(words[SIG_sigmaFreq_POS])
        except ValueError:
            fo.write('ERROR:readHistoSigNodeTable: invalid binFreq integer value \'' +
                     words[SIG_sigmaFreq_POS] + '\' on line ' +
                     str(line_no) + '\n')
            sys.exit(1)
            
        binFreq = iVal

        
        if  not nodeKey in nodeHistTable:
            nodeHistTable[nodeKey] = {}

        if binId in nodeHistTable[nodeKey]:
                fo.write('ERROR:readHistoSigNodeTable: invalid duplicate binId ' +
                         str(binId) +
                         ' on line ' +
                         str(line_no) +
                         ' in edge hist table ' +
                         '\n')
                sys.exit(1)
                
        nodeHistTable[nodeKey][binId] = binFreq

        if nodeKey in nodeFreqTable:
            nodeFreqTable[nodeKey] = nodeFreqTable[nodeKey] + binFreq
        else:
            nodeFreqTable[nodeKey] = binFreq

        line = fd.readline()

    fd.close()

    #if opt_print_edge_freq:
    if options.getStrConfig('opt_print_node_freq', fe) == 'True':

        fo.write('NODE FREQ --->\n')
        for curNode in sorted(nodeHistTable):
        
            fo.write('NODE: ' +
                     str(curNode) +
                     ' ' +
                     str(nodeFreqTable[curNode]) +
                     '\n')
        
            for curBinId in nodeHistTable[curNode]:
            
                fo.write('\tBIN: ' +
                         str(curBinId) +
                         ' ' +
                         str(nodeHistTable[curNode][curBinId]) +
                         '\n')
                
        fo.write('<--- NODE FREQ\n')
        fo.write('\n')

    
    #
    # integrity check for nodes and classes
    #

    #
    # all nodes belonging to the same output class,
    # must have the same node frequencies that is taken as output class frequency
    #

    classFreqTable = {}

    for curNode in nodeHistTable:

        if curNode[NODE_KEY_classId_POS] in classFreqTable:

            if classFreqTable[curNode[NODE_KEY_classId_POS]] != nodeFreqTable[curNode]:

                fo.write('ERROR: inconsistent class freq table ' +
                         str(classFreqTable[curNode[NODE_KEY_classId_POS]]) +
                         ' and node freq table ' +
                         str(nodeFreqTable[curNode]) +
                         'for node '+
                         str(curNode) +
                         '\n')
                sys.exit(1)

        else:
            classFreqTable[curNode[NODE_KEY_classId_POS]] = nodeFreqTable[curNode]

    #print(classFreqTable, '\n')

    #if opt_print_class_freq:
    if options.getStrConfig('opt_print_class_freq', fe) == 'True':

        fo.write('CLASS FREQ --->\n')
        for curClass in classFreqTable:

            fo.write('CLASS: ' +
                     str(curClass) +
                     ' ' +
                     str(classFreqTable[curClass]) +
                     '\n')
        
        fo.write('<--- CLASS FREQ\n')
        fo.write('\n')



    #
    # node freq table can be dropped (for memory space sake)
    # and use class freq table
    #

    nodeFreqTable = None


    

    #
    # estimate probabilities
    #

    #edgeProbTable = {}
    parNodeProbTable.clear()

    for curNode in nodeHistTable:

        if not curNode[NODE_KEY_classId_POS] in classFreqTable:
        
            fo.write('ERROR: estimate_probabilities: missing class ' +
                     curNode[NODE_KEY_classId_POS] +
                     ' from node ' +
                     str(curNode) +
                     ' in class freq table' +
                     '\n')
            sys.exit(1)

        if classFreqTable[curNode[NODE_KEY_classId_POS]] <= 0:
            fo.write('ERROR: estimate_probabilities: invalid null or negative class freq for class ' +
                     curNode[NODE_KEY_classId_POS] +
                     ' from node ' +
                     str(curNode) +
                     '\n')
            sys.exit(1)

        if not curNode in parNodeProbTable:
            parNodeProbTable[curNode] = {}

        for curBinId in nodeHistTable[curNode]:

            if not curBinId in parNodeProbTable[curNode]:
                parNodeProbTable[curNode][curBinId] = {}

            parNodeProbTable[curNode][curBinId] = (float(nodeHistTable[curNode][curBinId]) /
                                                float(classFreqTable[curNode[NODE_KEY_classId_POS]]))


    #if opt_print_edge_prob:
    if options.getStrConfig('opt_print_node_prob', fe) == 'True':
#
#        fo.write('NODE PROB --->\n')
#        for curNode in sorted(nodeHistTable):
#        
#            fo.write('NODE: ' +
#                     str(curNode) +
#                     ' ' +
#                     #str(nodeFreqTable[curNode]) +
#                     str(classFreqTable[curNode[NODE_KEY_classId_POS]]) +
#                     '\n')
#        
#            for curBinId in nodeHistTable[curNode]:
#            
#                fo.write('\tBIN: ' +
#                         str(curBinId) +
#                         ' ' +
#                         str(sigmaLbTable[curNode][curBinId]) +
#                         ' ' +
#                         str(sigmaUbTable[curNode][curBinId]) +
#                         '\t' +
#                         str(nodeHistTable[curNode][curBinId]) +
#                         '\t' +
#                         str(round(parNodeProbTable[curNode][curBinId], F_PREC)) +
#                         '\n')
#            
#        fo.write('<--- NODE PROB\n')
#        fo.write('\n')
#
        
        printNodeProbTable(nodeHistTable,
                           classFreqTable,
                           sigmaLbTable,
                           sigmaUbTable,
                           parNodeProbTable,
                           options)
        
    return


def printNodeProbTable(nodeHistTable,
                       classFreqTable,
                       sigmaLbTable,
                       sigmaUbTable,
                       parNodeProbTable,
                       options):
    
    #
    # be careful to keep this field POS
    # consistent with readHistoSigNodeTable
    #
    
    NODE_KEY_classId_POS = 2

    fo.write('NODE PROB --->\n')
    for curNode in sorted(nodeHistTable):
        
        fo.write('NODE: ' +
                 str(curNode) +
                 ' ' +
                 #str(nodeFreqTable[curNode]) +
                 str(classFreqTable[curNode[NODE_KEY_classId_POS]]) +
                 '\n')
        
        for curBinId in nodeHistTable[curNode]:
            
            fo.write('\tBIN: ' +
                     str(curBinId) +
                     ' ' +
                     str(sigmaLbTable[curNode][curBinId]) +
                     ' ' +
                     str(sigmaUbTable[curNode][curBinId]) +
                     '\t' +
                     str(nodeHistTable[curNode][curBinId]) +
                     '\t' +
                     str(round(parNodeProbTable[curNode][curBinId], F_PREC)) +
                     '\n')
            
    fo.write('<--- NODE PROB\n')
    fo.write('\n')

    return
    
