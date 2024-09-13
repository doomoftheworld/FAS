#!/usr/bin/python3

import sys
import string
import re

class configCl:

    #
    # class member accessible as configCl._configSet
    #_configSet = {}
    #

    #
    # standard constructor
    #

    def __init__(self):
        self._configSet = {}

    def init(self):
        self._configSet = {}
                      
    def readConfig(self, finput, fe, fo):

        #
        # reads selection id's
        #

        line_no = 0
        self._configSet.clear()

        line = finput.readline()
        while line != "":

            line_no = line_no + 1
            #fo.write('SELECTION LINE NO:' + `line_no` + '\n')
            #fo.write('SELECTION LINE:' + line + '\n')
            line = re.sub('\n$', '', line)


            #
            # skip empty or comment lines
            #

            if (line == '') or (re.match('^#', line)):
                line = finput.readline()
                continue

            #words = string.split(line)
            words = line.split()

            if len(words) != 2:
                fe.write('ERROR: inconsistent config length ' +
                         str(len(words)) +
                         ' at line ' +
                         str(line_no) + '\n')
                sys.exit(1)

            #fo.write('LINE: ' + line + '\n')
            #fo.write('LENGTH: ' + `len(words)` + '\n')
            #print words

            #if self._configSet.has_key(words[0]):
            if words[0] in self._configSet:
                fe.write('ERROR: duplicate config ' +
                         words[0] + '\n')
                sys.exit(1)

            self._configSet[words[0]] = words[1]

            line = finput.readline()

            
    def readFileNameConfig(self, fileNameStr, fe, fo):

        finput = open(fileNameStr, 'r')

        #
        # reads selection id's
        #

        line_no = 0
        self._configSet.clear()

        line = finput.readline()
        while line != "":

            line_no = line_no + 1
            #fo.write('SELECTION LINE NO:' + `line_no` + '\n')
            #fo.write('SELECTION LINE:' + line + '\n')
            line = re.sub('\n$', '', line)


            #
            # skip empty or comment lines
            #

            if (line == '') or (re.match('^#', line)):
                line = finput.readline()
                continue

            #words = string.split(line)
            words = line.split()

            if len(words) != 2:
                fe.write('ERROR: inconsistent config length ' +
                         str(len(words)) +
                         ' at line ' +
                         str(line_no) + '\n')
                sys.exit(1)

            #fo.write('LINE: ' + line + '\n')
            #fo.write('LENGTH: ' + `len(words)` + '\n')
            #print words

            #if self._configSet.has_key(words[0]):
            if words[0] in self._configSet:
                fe.write('ERROR: duplicate config ' +
                         words[0] + '\n')
                sys.exit(1)

            self._configSet[words[0]] = words[1]

            line = finput.readline()
            
        finput.close()

                                            
    def printConfig(self, fe, fo):

        #kList = self._configSet.keys()
        #kList.sort()

        kList = sorted(self._configSet)
        
        fo.write('CONFIG --->\n')
        for k in kList:
            fo.write('\t' + k + ' ' + self._configSet[k] + '\n')
        fo.write('<--- CONFIG\n')
        fo.write('\n')


    def hasOption(self, str):

        #if self._optionSet.has_key(str):
        if str in self._optionSet:
            return 'True'
        else:
            return 'False'

        
    def getOption(self, str):

        #if self._configSet.has_key(str):
        if str in self._optionSet:
            return self._configSet[str]
        else:
            return None


    def getStrictStrConfig(self, str, fe):

        #if not self._configSet.has_key(str):
        if not str in self._configSet:
            fe.write('ERROR: missing ' + str + ' configuration\n')
            sys.exit(1)

        return self._configSet[str]


    def getStrictIntConfig(self, str, fe):

        #if not self._configSet.has_key(str):
        if not str in self._configSet:
            fe.write('ERROR: missing ' + str + ' configuration\n')
            sys.exit(1)

        try:
            iVal = int(self._configSet[str])
        except ValueError:
                        fe.write('ERROR: invalid configuration integer ' +
                                 self._configSet[str] +
				 ' for key ' +
				 str +
                                 '\n')
                        sys.exit(1)

        return iVal

    def getStrictFloatConfig(self, str, fe):

        #if not self._configSet.has_key(str):
        if not str in self._configSet:
            fe.write('ERROR: missing ' + str + ' configuration\n')
            sys.exit(1)

        try:
                        rVal = float(self._configSet[str])
        except ValueError:
                        fe.write('ERROR: invalid configuration float ' +
                                 self._configSet[str] +
				 ' for key ' +
				 str +
                                 '\n')
                        sys.exit(1)

        return rVal

    
    def getStrConfig(self, str, fe):

        if not str in self._configSet:
            return None

        return self._configSet[str]

    def getIntConfig(self, str, fe):

        if not str in self._configSet:
            return None

        try:
            iVal = int(self._configSet[str])
        except ValueError:
                        fe.write('ERROR: invalid configuration integer ' +
                                 self._configSet[str] +
				 ' for key ' +
				 str +
                                 '\n')
                        sys.exit(1)

        return iVal

    def getFloatConfig(self, str, fe):

        if not str in self._configSet:
            return None

        try:
                        rVal = float(self._configSet[str])
        except ValueError:
                        fe.write('ERROR: invalid configuration float ' +
                                 self._configSet[str] +
				 ' for key ' +
				 str +
                                 '\n')
                        sys.exit(1)

        return rVal

options = configCl()
config = configCl()
