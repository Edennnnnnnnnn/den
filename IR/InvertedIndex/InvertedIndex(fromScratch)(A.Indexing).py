"""
By Eden Zhou
Feb. 08, 2023
"""

import re
import sys
import csv
import json

from nltk.tokenize import word_tokenize

class Indexing:
    def __init__(self):
        self.outputDirectory = None

        self.allData = None
        self.allZones = None

    def initializer(self, inputCorpusFile, outputIndexesDirectory):
        """
            This is a setter function that is used to add data file paths to the operator and provide initial processing;
        Data are loaded based on JSON format and stored in the operator;
        :param:
            - inputCorpusFile, str, input path read from the command line, be used to locate input file location;
            - outputIndexesDirectory, str, output path read from the command line, be used to locate output directory location;
        """
        self.outputDirectory = outputIndexesDirectory
        # Data are loaded based on JSON format and stored in the operator:
        with open(inputCorpusFile, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
            self.allData = data

    def outputWriter(self, outputDirectoryPath: str, outputFileName: str, outputData: list) -> None:
        """
            The overall control function for the whole model project;
        :param:
            - outputDirectoryPath, str, Inputs from the command line, be used to locate output directory path;
            - outputFileName, str, based on "zone names", is used to specify the output file name;
            - outputData, list, the after-indexing dataset that will be stored into the output .tsv file;
        """
        # Based on given paths and names, combining to get the full output path:
        outputPathFile: str = outputDirectoryPath + outputFileName + ".tsv"

        # Printing header and output data to specific .tsv files:
        header: list = ['token', 'DF', 'postings']
        with open(outputPathFile, 'w', newline='', encoding="utf8") as outputFile:
            writer = csv.writer(outputFile, delimiter="\t")
            writer.writerow(header)
            writer.writerows(outputData)

    def indexing(self) -> None:
        """
            The method is used to process the indexing and rearrange the data structure, more information can be found
        in README.md file under the program package; Recall Json raw data structure: [{docID:XXX, zone1: XXX, zone2:XXX},
        {docID:XXX, zone1: XXX, zone2:XXX}]; and data structure in processing: {zonename_1:{token_1:[docID_1, docID_2,
        ...], token_2:[docID_3, docID_4, ...], ...},zoneName_2:{token_i:[docID_i, ...]}, ...};
        """
        idChecker = []
        allZones = {}
        # Processing indexing and error handling measures:
        for doc in self.allData:
            if "doc_id" in doc.keys():
                currDoc = doc["doc_id"]
                if currDoc in idChecker:
                    sys.stderr.write("\n>> DuplicateDocIDError: The corpus input contains multi-docs with the same docID;")
                    exit()
                idChecker.append(currDoc)
                del doc['doc_id']
            else:
                sys.stderr.write("\n>> NoDocIDError: One or more datalines provided have no data_id that can be processed;")
                exit()
            if len(doc.keys()) == 0:
                sys.stderr.write(f"\n>> NoZoneFoundError: One or more datalines provided under docID={currDoc} have no zones existed; Reminder: Each document should have at least one zone with some text associated;")
                exit()
            for currZone in doc.keys():
                checker = ''.join(re.findall("[A-Za-z0-9]", currZone))
                if checker != currZone:
                    sys.stderr.write(f"\n>> IllegalZoneName: One or more datalines provided under docID={currDoc} have illegal zone name; Reminder: no space or punctuation are allowed in a zone name;")
                    exit()
                tokens = list(set(word_tokenize(doc[currZone].lower())))
                if currZone not in allZones:
                    allZones[currZone] = {}
                for token in tokens:
                    if token in allZones[currZone]:
                        allZones[currZone][token].append(currDoc)
                        sorted(allZones[currZone][token])
                    else:
                        allZones[currZone][token] = [currDoc]

        # Rebuilding the output format for output:
        for zone in allZones.keys():
            review = []
            tks = allZones[zone].keys()
            for tk in tks:
                review.append([tk, len(allZones[zone][tk]), ', '.join(allZones[zone][tk])])
            # Call the output function to write:
            self.outputWriter(self.outputDirectory, zone, sorted(review))


def main(argv, operator=Indexing()):
    """
        The overall control function for the whole model project;
    :param:
        - argv, list, Inputs from the command line, be used to locate data input and output paths;
        - operator, Indexing, one object of the Indexing class, which will be used for processing the tasks;
    """

    """ [Init] Handling all command parameters entered & Initializing the predictor """
    # Read from commandline arguments.
    try:
        inputCorpusFile = argv[1]
        outputIndexesDirectory = argv[2]
    # Checking errors in command entered, if TRUE, raise an error and quit:
    except IndexError:
        # inputCorpusFile = "/Users/den/Desktop/CMPUT 361 - A1/w23-hw1-Edennnnnnnnnn/data/dr_seuss_lines.json"
        # outputIndexesDirectory = "/Users/den/Desktop/CMPUT 361 - A1/w23-hw1-Edennnnnnnnnn/output/"
        sys.stderr.write("\n>> CommandError: Invalid command entered, please check the command and retry;")
        exit()
    if outputIndexesDirectory[-1] != '/':
        outputIndexesDirectory += '/'

    """ [Input] Acquire the data input from the path provided """
    operator.initializer(inputCorpusFile, outputIndexesDirectory)

    """ [Processing] Processing the indexing tasks based on the given data """
    operator.indexing()


if __name__ == "__main__":
    main(sys.argv)
    