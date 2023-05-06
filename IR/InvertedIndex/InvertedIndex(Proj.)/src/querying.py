import sys
import csv
import os

def pCheker(pStack) -> bool:
    """
        This is a helper function that is used to recheck the logical correctness of paired contexts parsed, if any error
    exists, False will be returned to stop the whole parsing process;
    :return:
        - logicalCorrectness, boolean, False for any possible error existed in the query input;
    """
    paired = []
    if len(pStack) < 2:
        return False
    else:
        for i in pStack:
            if len(paired) == 0:
                paired.append(i)
            elif i == "(":
                paired.append(i)
            else:
                if paired[-1] == "(" and i == ')':
                    paired.pop()
                else:
                    return False
        if len(paired) == 0:
            return True
        else:
            return False


class Querying:
    def __init__(self):
        self.query = None
        self.zones = {}
        self.allTokens = []

    def setIndexes(self, indexesDirPath):
        """
            Read and store all tokens and corresponding DocID indexed based on the directory path provided;
        :param:
            - indexesDirPath, str, the input path that pointing to the directory of indexing files;
        """
        for name in self.zones:
            currList = [name]
            currDict = {}
            with open(indexesDirPath + name + '.tsv', "r") as file:
                reader = csv.DictReader(file, delimiter="\t")
                for line in reader:
                    currDict[line['token']] = set(map(int, line['postings'].split(', ')))
            currList.append(currDict)
            self.allTokens.append(currList)

    def setQuery(self, queryInput):
        """
            Analyzing and parsing the query input piece by piece, the logical correctness and querying validation will
        be checked in this stage. If all looks good then store all partial queries for further operations. Otherwise,
        printing an error message and stopping the program;
        :param:
            - queryInput, str, the input query that will be used for processing;
        """
        self.query = queryInput.split()
        pStack = []
        keyword = ['AND', 'OR']

        # Check parenthesis and store them separately
        for i, item in enumerate(self.query):
            # Case 1: Contain many pairs of '()'
            if '(' in item and ')' in item and (item.count('(') > 1 or item.count(')') > 1):
                sys.stderr.write(">> Query format usage of parenthesis '(' and ')' is invalid!")
                exit()

            # Case 2: '(xxx)', 'x(x)x', ')xxx(', 'x)x(x'
            elif '(' in item and ')' in item and len(item) > 2:
                if (item[0] == '(' and item[-1] == ')') or (item[-1] == '(' and item[0] == ')'):
                    self.query[i] = item[1:-1]
                else:
                    sys.stderr.write(">> Query format usage of parenthesis '(' and ')' is invalid!")
                    exit()

            # Case 3: '(xxx' and 'x(xx'
            elif '(' in item and len(item) > 1:
                if item[0] == '(' and item.count('(') == 1 and item[1:] not in keyword:
                    pStack.append('(')
                    self.query[i] = item[1:]
                    self.query.insert(i, '(')
                elif item.count('(') > 1:
                    currL = i
                    while item[0] == '(':
                        pStack.append('(')
                        if i == 0:
                            self.query.insert(0, '(')
                            currL += 1
                        else:
                            self.query.insert(i - 1, '(')
                            currL -= 1
                        item = item[1:]
                    self.query[currL] = item
                else:
                    sys.stderr.write(">> Query format usage of parenthesis '(' is invalid!")
                    exit()

            # Case 4: 'xxx)' and 'x)xx'
            elif ')' in item and len(item) > 1:
                if item[-1] == ')' and item.count(')') == 1 and item[:-1] not in keyword:
                    pStack.append(')')
                    self.query[i] = item[:-1]
                    if i + 1 == len(self.query):
                        self.query.append(')')
                    else:
                        self.query.insert(i, ')')
                elif item.count(')') > 1:
                    currR = i
                    while item[-1] == ')':
                        pStack.append(')')
                        if i + 1 == len(self.query):
                            self.query.append(')')
                            currR -= 1
                        else:
                            self.query.insert(i + 1, ')')
                            currR += 1
                        item = item[:-1]
                    self.query[currR] = item
                else:
                    sys.stderr.write(">> Query format usage of parenthesis ')' is invalid!")
                    exit()

        # Check the pairs
        if pStack and not pCheker(pStack):
            sys.stderr.write(">> Query format usage of parenthesis is invalid!")
            exit()

        # Check the keyword
        for i in range(len(self.query) - 1):
            # and, or, and not, or not, not and, not or, or and
            if self.query[0].upper() in keyword or self.query[-1].upper() in keyword:
                sys.stderr.write(">> Query format (usage of keyword) is invalid!")
                exit()
            elif self.query[i].upper() in keyword and self.query[i + 1].upper() in keyword:
                sys.stderr.write(">> Query format (usage of keyword) is invalid!")
                exit()
            elif self.query[i].upper() == 'NOT' and self.query[i + 1].upper() in keyword:
                sys.stderr.write(">> Query format (usage of keyword) is invalid!")
                exit()
            elif i == 0 and self.query[i].upper() == 'NOT':
                sys.stderr.write(">> Query format (usage of keyword) is invalid!")
                exit()
            elif self.query[i - 1].upper() not in keyword and self.query[i].upper() == 'NOT' and self.query[i + 1].upper() not in keyword:
                sys.stderr.write(">> Query format (usage of keyword) is invalid!")
                exit()

        # Check the zone name
        for i, item in enumerate(self.query):
            self.query[i] = item.lower()
            if ':' in item:
                item = item.split(':')
                if item[0] not in self.zones and len(item) != 2:
                    sys.stderr.write(">> Query format (zone name not exist) is invalid!")
                    exit()
            else:
                if item.upper() not in keyword and item.upper() != 'NOT' and item.isalpha():
                    sys.stderr.write(">> Query format is invalid! Please enter in form 'zone:token'")
                    exit()

        # Check is token exist and substitute each token by their sets
        for i, item in enumerate(self.query):
            if ':' in item:
                item = item.split(':')
                try:
                    if item[1] not in self.allTokens[self.zones[item[0]]][1]:
                        self.query[i] = str(set())
                    else:
                        self.query[i] = str(self.allTokens[self.zones[item[0]]][1][item[1]])
                except KeyError:
                    sys.stderr.write(f">> Invalid Zone Name, the database doesn't contain zone '{item[1]}'")
                    exit()

    def parsingQuery(self):
        """
            Parsing the partial queries processed before and expressing the query signs to symbol signs,
        """
        # Substitute operators
        for i, item in enumerate(self.query):
            if i + 1 < len(self.query):
                if self.query[i] == 'and' and self.query[i + 1] == 'not':
                    self.query[i] = '-'
                    del self.query[i + 1]
                elif item == 'or': self.query[i] = '|'
                elif item == 'and': self.query[i] = '&'

        check = 0

    def output(self):
        """
            The output function of the querying process. Recombining the query parsed in symbol format, evaluating and
        finding output the target data indexed, writing results to the stdout stream;
        """
        # Evaluate Query
        query = " ".join(self.query)
        result = eval(query)
        for i in result:
            sys.stdout.write(str(i) + "\n")


def main(argv, operator=Querying()):
    """
        The overall control function for the whole model project;
    :param:
        - argv, list, Inputs from the command line, be used to locate data input and output paths;
        - operator, Querying, The object for querying process;
    """
    """ [Initializing] Handling all command parameters entered & Initializing the predictor """
    # Setting for Terminal Running:
    try:
        indexesDirPath = argv[1]
        booleanQuery = argv[2]
    # Setting for Manual Running:
    except IndexError:
        # indexesDirPath = "/Users/den/Desktop/CMPUT 361 - A1/w23-hw1-Edennnnnnnnnn/output/"
        # booleanQuery = "book:Lorax OR (book:you AND line:you)"
        sys.stderr.write(">> CommandError: Invalid command entered, please check the command and retry;")
        exit()

    """ [Input] Acquire the indexing data input from the path provided and pre-processing the query"""
    # Store the file names
    i = 0
    for inputFileName in os.listdir(indexesDirPath):
        if '.tsv' in inputFileName:
            name = inputFileName[:-4]
            operator.zones[name] = i
            i += 1
    operator.setIndexes(indexesDirPath)

    if booleanQuery == "":
        sys.stderr.write(">> Query is empty!")
        exit()
    operator.setQuery(booleanQuery)

    """ [Processing] Parsing the query given and processing the querying task """
    operator.parsingQuery()

    """ [Output] Writing the querying results to STDOUT """
    operator.output()


if __name__ == "__main__":
    main(sys.argv)
