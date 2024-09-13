#!/usr/bin/python3

import sys
import math
import re
from sigConstants import *

#from config3Cl import options

fi = sys.stdin
fo = sys.stdout
fe = sys.stderr

FILENAME_EXPECTED_LINE_LEN = 1

def readClassList(classListFileName, options):

    classList = []

    fd = open(classListFileName, 'r')

    line_no = 0
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
        if options.getStrConfig('opt_debug_read_class_list', fe) == 'True':
            fo.write('LINE: ' + line + '\n')
            fo.write('LENGTH: ' + str(len(words)) + '\n')
            #print words
        
        if len(words) != FILENAME_EXPECTED_LINE_LEN:
            fo.write('ERROR: readClassList: invalid file list line length ' +
                     str(len(words)) +
                     ' at line ' +
                     str(line_no) +
                     ' (' +
                     str(FILENAME_EXPECTED_LINE_LEN) +
                     ' expected)' +
                     '\n')
            sys.exit(1)

        if options.getStrConfig('opt_debug_read_class_list', fe) == 'True':
            fo.write('WORDS --->\n')
            i = 0
            while i < len(words):
                fo.write('\tWORD ' + str(i) + ': ' + words[i] + '\n')
                i = i + 1
            
            fo.write('<--- WORDS\n')
            fo.write('\n')

        classList.append(words[0])

        line = fd.readline()

    fd.close()

    return classList
