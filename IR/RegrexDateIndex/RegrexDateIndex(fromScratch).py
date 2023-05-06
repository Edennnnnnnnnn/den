import re
import os
import sys
import csv
from nltk.tokenize import word_tokenize


def regexFilter(validMatches):
    """
    :param:
        - validMatches: List[][], 2d, [['article_id', 'expr_type', 'value', "offset"], [...], ..., [...]],
    :return
        - validMatchesSimplified: List[][], 2d, [['article_id', 'expr_type', 'value', "offset"], [...], ..., [...]],
    """
    # Remove Useless Types:
    validMatchesSimplified, validMatchesDuplicateRemoved = [], []
    for dataline in validMatches:
        if (dataline[1] == 'relatives') or (dataline[1] == "num_lessThanHundr") or (dataline[1] == "englnum") or (dataline[1] == "englidx") or (dataline[1] == "partials") or (dataline[1] == "partials") or (dataline[1] == "timeUnit") or (dataline[1] == 'quarter'):
            continue
        else:
            validMatchesSimplified.append(dataline)
            validMatchesDuplicateRemoved.append(dataline)

    # Remove Short Duplicates:
    for i in validMatchesSimplified:
        for j in validMatchesSimplified:
            isFileSame = (i[0] == j[0])
            isOneShorter = (len(i[2]) > len(j[2]))
            isContainRelationWorks = ((j[2] in word_tokenize(i[2])) or (j[2] in i[2]))
            isDistanceValid = abs(i[3] - j[3]) < 20
            if i is j:
                continue
            elif isFileSame and isOneShorter and isContainRelationWorks and isDistanceValid:
                if j in validMatchesDuplicateRemoved:
                    validMatchesDuplicateRemoved.remove(j)

    return validMatchesDuplicateRemoved


def regexRelationMerging(mergingNeededDatalines):
    """
        Put two or more connected tokens together to form a new one with combined value/type marked;
    :param:
        - scopeMerging: List[][], 2d, [['article_id', 'expr_type', 'value', (scope)], [...], ..., [...]], contains all
        neighbor scopes that should be merged to one large scope, shown in data line provided;
    :return
        - scopeMerged: List[],  ['article_id', 'expr_type', 'value', (scope)], deconstructing scopes and relevant data
        inputted and recreate a single combined (merged) data line in the list;
    """

    # *** Second-level Regex Matching Rule ***
    # 2 regexesRel_month-year = (month + year)
    # 2 regexesRel_part-of-decade = (relatives + decade)
    # 2 regexesRel_month-to-month = (month + month)
    # 2 regexesRel_num-of-decades/years/months/weeks/days = (englnum + timeUnit)
    # 3 regexesRel_relative-years/months/weeks/days = (relatives + englnum/num_lessThanHundr + timeUnit)
    # 3 regexesRel_part-of-relative-year/month/week/day = (partials + relatives + timeUnit)
    # 3 regexesRel_quarter-year = (englidx/relatives + quarter + year)
    # 3 regexesRel_relative-month&year = (month + relatives + timeUnit)

    # Deconstruct all existed dataline matches:
    typesInput = []
    valuesInput = []
    scopesInput = []
    for dataline in mergingNeededDatalines:
        typesInput.append(dataline[1])
        valuesInput.append(dataline[2])
        scopesInput.append(dataline[3])

    # Checking the relations between two or more tokens, if the relation is valid, a new type will be applied:
    typesInputNum = len(typesInput)
    new_type = None
    if typesInputNum == 2:
        if (typesInput[0] == "month") and (typesInput[1] == "year"):
            new_type = "month-year"
        elif (typesInput[0] == "relatives") and (typesInput[1] == "decade"):
            new_type = "part-of-decade"
        elif typesInput.count("month") == 2:
            new_type = "between-months"
        elif (typesInput[0] == "relatives") and (typesInput[1] == "timeUnit"):
            new_type = "relative-year/month/week"
        elif (("englnum" in typesInput) or ("num_lessThanHundr" in typesInput)) and ("timeUnit" in typesInput):
            new_type = "num-of-decades/years/months/weeks/days"
    elif typesInputNum == 3:
        if (typesInput[0] == "relatives") and ((typesInput[1] == "englnum") or (typesInput[1] == "num_lessThanHundr")) and (typesInput[2] == "timeUnit"):
            new_type = "relative-year/month/week"   # "relative-years/months/weeks/days"
        elif (typesInput[0] == "partials") and (typesInput[1] == "relatives") and (typesInput[2] == "timeUnit"):
            new_type = "part-of-relative-year/month/week"
        elif ((typesInput[0] == "englidx") or (typesInput[0] == "relatives")) and (typesInput[1] == "quarter") and (typesInput[2] == "year"):
            new_type = "quater-year"
        elif (typesInput[0] == "month") and (typesInput[1] =="relatives") and (typesInput[2] == "timeUnit"):
            new_type = "month, relative-year"

    # Rebuild the merged datalines matches, if a valid new type found, then return it as a successful merging case:
    new_value = " ".join(valuesInput)
    new_scope = (scopesInput[0][0], scopesInput[-1][-1])
    datalineMerged = [mergingNeededDatalines[0][0], new_type, new_value, new_scope]
    if datalineMerged[1] is not None:
        return datalineMerged
    return


def regexRelationChecking(validMatches) -> list:
    """
        Checking words relation, if the neighbor token of the select token is also selected by the Regex matcher, then
    they are marked as relational, then put them together to form a new token with combined type/value;
    :param:
        - validMatches: List[][], 2d, [['article_id', 'expr_type', 'value', (scope))], [...], ..., [...]];
    :return
        - validMatches_relationChecked: List[][], 2d, [['article_id', 'expr_type', 'value', (scope))], [...], ..., [...]];
    """
    # Build the data structure (dict) for storing all existed dataline matches:
    hashTable = {}
    scopes = []
    for match in validMatches:
        scopeMatched = match[3]
        hashTable[scopeMatched] = match
        scopes.append(scopeMatched)
    scopes.sort()

    # Checking the matched tokens, if token matched exists close to each other, then they are treated as a group, which
    #   will be added to a list for further merging;
    currentIndex = 0
    nextIndex = 1
    newScopes = []
    scopesNum = len(scopes)
    while currentIndex < scopesNum:
        currentDataLine = hashTable.get(scopes[currentIndex])
        mergingNeededDatalines = [currentDataLine]
        isMergingNeeded = False
        while (nextIndex < scopesNum) and (scopes[currentIndex][1] + 1 == scopes[nextIndex][0]):
            isMergingNeeded = True
            nextDataLine = hashTable.get(scopes[nextIndex])
            mergingNeededDatalines.append(nextDataLine)
            currentIndex = nextIndex
            nextIndex = currentIndex + 1
        if not isMergingNeeded:
            currentIndex = nextIndex
            nextIndex = currentIndex + 1
        if (currentIndex or nextIndex) > len(scopes):
            break

        # Merging the relational group of matched datalines:
        if isMergingNeeded:
            sorted(mergingNeededDatalines)
            datalineAfterMerging = regexRelationMerging(mergingNeededDatalines)
            if datalineAfterMerging is not None:
                newScopes.append(datalineAfterMerging)
        else:
            newScopes.append(currentDataLine)
    return newScopes


def regexAnalyzing(matcher) -> list:
    """
        Access to the outputs from re.finditer() and reconstruct them as a list type object, as [int, (int, int)];
    :param:
        - matcher: Iterator, based on a specific regex pattern for a given text, as data returned from re.finditer();
    :return
        - targets: List, 2d, [['value', (scope)], ...], as pattern searching results for a given text;
    """
    # Properly read the output from re.finditer() and build up the basic data structure:
    feedback = []
    for target in matcher:
        scopeTuple = (target.start(), target.end())
        feedback.append([target.group(), scopeTuple])
    return feedback


def scopeToIndexes(inputScope) -> list:
    """
        Transfer the scope expression within a tuple to indexes expression in a list object;
    :param:
        - inputScope: Tuple, (startIndex, endIndex)
    :return
        - outputIndexes: List, [startIndex, startIndex+1, ..., endIndex-1, endIndex]
    """
    # print("inputScope", inputScope)
    # Changing scope expression to index expression, for easier duplicates checking later:
    firstIndex = inputScope[0]
    lastIndex = inputScope[1]
    outputIndexes = []
    while firstIndex < lastIndex:
        outputIndexes.append(firstIndex)
        firstIndex += 1
    return outputIndexes


def scopeToOffset(validMatches_checked) -> list:
    """
        Accept datalines with scope expression, change it to the num of its first index, then output edited datalines;
    :param:
        - validMatches_checked: list, 2d, contains all datalines with scope expressions;
    :return
        - validMatches_checked: list, 2d, contains all datalines with only the index of the first element shown;
    """
    # Changing dataline matches with scope expression to matches with offset expression:
    for index in range(len(validMatches_checked)):
        firstCharIndex = validMatches_checked[index][3][0]
        validMatches_checked[index][3] = firstCharIndex
    return validMatches_checked


def regexMatching(text, regexesLexicon, inputFile) -> list:
    """
        Searching the text input based on the given regexesLexicon, all matching data will be reformed as a 2d list with
    all relative data needed for some further operations;
    :param:
        - text: List[String], as the text in file given for pattern matching;
        - regexesLexicon: Dictionary{String, List[String]}, contains all possible regex patten for matching;
        - inputFile: String, as the name of the text file;
    :return
        - validMatches_relationChecked: List[][], 2d, [['article_id', 'expr_type', 'value', 'offset'], [...], ..., [...]];
    """
    indexMap = []
    validMatches = []
    # Apply regexes to text input:
    for regexType, regex in regexesLexicon.items():
        isMatcherExisted = re.search(regex, text)
        if isMatcherExisted is not None:
            matches = re.finditer(regex, text)
            # Read the matching results and reshape the data expressions:
            feedbacks = regexAnalyzing(matches)
            for feedback in feedbacks:
                indexesTaken = scopeToIndexes(feedback[1])
                # Specific Format Double-check:
                if regexType == "year":
                    # to clean the space and punctuation in the value of "year" type 
                    opt_feedback = regexAnalyzing(re.finditer("\d{3,4}", feedback[0]))[0]
                    feedback[0] = opt_feedback[0]
                    feedback[1] = (feedback[1][0] + opt_feedback[1][0], feedback[1][0] + opt_feedback[1][1])
                # Check for Index Duplicates, if no duplicates appeared, then store the new dataline:
                if indexesTaken not in indexMap:
                    validMatches.append([inputFile, regexType] + feedback)
                    indexMap += indexesTaken
    # Do relational checking, words merging and useless tags removing:
    validMatches_checked = scopeToOffset(regexRelationChecking(validMatches))
    validMatches_simplified = regexFilter(validMatches_checked)
    return validMatches_simplified


def main(argv):
    """
        The controller function of the whole program;
    :param:
        - argv, list, Inputs from the command line, be used to locate data input and output pathes;
    """
    """ 
    RegexLexicon:
        All possible regexes patterns that stored in diverse types for different words matching;
    """
    monthName = "(January|February|March|April|May|June|July|August|September|October|November|December)"   # Match one of the twelve Months
    regexes_dayofweek = "(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day"  # Match one of the day during a week
    regexes_day_month_year = "\d{1,2}\s" + monthName + ",?\s\d{3,4}"    # day(1-2 digits) Month with comma or not and year(3-4 digits)
    regexes_month_year = monthName + "\s\d{3,4}"    # Month with year(3-4 digits)
    regexes_month_day = monthName + "\s[1-3]?(1(st)?|2(nd)?|3(rd)?|\d(th)?)" # Month with date(1-2 digit) with or without ordinal numbers
    regexes_day_month = "\d{1,2}\s" + monthName     # data(1-2 digit) Month
    regexes_month = monthName   # Match one of the twelve Months
    regexes_relatives = "([Tt]he\s)?(last|this|next)"   # to show one of the three relations with The or the in begin
    regexes_partials = "(later|earlier)" # to show one of the two partials
    regexes_englnum = "([Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine|[Tt]en)"    # english number from 1-10
    regexes_englidx = "([Ff]irst|[Ss]econd|[Tt]hird|[Ff]ourth])"    # ordinal number 1-4 that prepare for quarter
    regexes_timeUnit = "([Dd]ecade|[Yy]ear|[Mm]onth|[Ww]eek(end)?|[Dd]ay)s?" # match with one of the different time unit
    regexes_quarter = "[Qq]uarter(\sof)?"   # Quarter with capital Q or not and with 'of' or not
    regexes_decade = "(the\s)?\d{3,4}s"     # eg. the 1990s
    regexes_year = "(^|[\"\s])[1-9]\d{3}[\.\,\?]?([\"\s])"  # four digit year with space or quote before and with period, comma, question mark, space, quote after
    regexes_lessThanHundr = "[1-9]\d?"  # numerical number from 1-99

    regexesLexicon = {"dayofweek": regexes_dayofweek,
                      "day-month-year": regexes_day_month_year,
                      "month-year": regexes_month_year,
                      "month-day": regexes_month_day,
                      "day-month": regexes_day_month,
                      "month": regexes_month,
                      "relatives": regexes_relatives,
                      "partials": regexes_partials,
                      "englnum": regexes_englnum,
                      "englidx": regexes_englidx,
                      "timeUnit": regexes_timeUnit,
                      "quarter": regexes_quarter,
                      "decade": regexes_decade,
                      "year": regexes_year,
                      "num_lessThanHundr": regexes_lessThanHundr
                      }

    """ 
    I/O Dashboard:
        Control the input and output setting, processing functions needed;
    """
    # Manually set the input/output path (for testing):
    inputDirectory = argv[1]
    outputFile = argv[2]

    # Sort/read all data input from files provided, apply matching operations:
    output = []
    dataList = sorted(os.listdir(inputDirectory))
    for inputFile in dataList:
        text = dataInput(inputDirectory, inputFile)
        output += regexMatching(text, regexesLexicon, inputFile)

    # Printing the matching results to the specific CSV file:
    dataOutput(outputFile, output)


def dataInput(inputDirectory, inputFile):
    """
        The data input/read function of the whole program;
    :param:
        - inputDirectory, str, the path of the directory which contains all data input files;
        - inputFile, str, the specific input file name;
    :return
        - text, str, the data (text) read from the file with filename provided;
    """
    # Open a specific data file and read all its contexts as a str:
    with open(inputDirectory + inputFile, "r") as file:
        text = file.read()
        return text


def dataOutput(outputFile, output):
    """
        The data output/print function of the whole program;
    :param:
        - outputFile, str, the path (name) of the target output file;
        - output, list, 2d, the data sturcture that stored all matching results;
    """
    # Printing all matching result (stored in output) to the specific path(file) provided:
    with open(outputFile, 'w', newline='') as outputFile:
        header = ['article_id', 'expr_type', 'value', 'char_offset']
        writer = csv.writer(outputFile)
        writer.writerow(header)
        writer.writerows(output)


if __name__ == "__main__":
    main(sys.argv)
