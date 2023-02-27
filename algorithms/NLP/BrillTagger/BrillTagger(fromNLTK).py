class BrillModel:
    def __init__(self):
        """ Class Variables Initialized"""
        # Model Parameter Settings:
        self.tagger = None
        self.baselineTagger = HMMModel()

        # Model Storages Settings:
        self.dataTrain = None
        self.dataTest = None
        self.dataTestTagged = None

    def modelInput(self, inputData: list, dataType: str = None) -> None:
        """
            This is the data input function for the whole program; Filling self.dataTrain for training or
        self.dataTest for tests; All data were read from the valid TXT file under the target directory and have been
        pre-processed by the file reader function;
        :param:
            - self
            - inputData, list, the preprocessed data input;
            - dataType, str, corresponds to two possible types, for differentiating input data type;
        """
        # Open specific data input files within the given directory path and record all their contexts:
        if dataType == "TRAIN":
            self.dataTrain = inputData
        elif dataType == "TEST":
            self.dataTest = inputData
        else:
            print("DataTypeError: Invalid input data type;")

    def modelTrain(self) -> None:
        """
            This is the training control method for the whole process; This function builds up the Brill template
        manually and then applying data for training; A baseline model of HMMs is also prepared for the BrillTrainer;
        :param:
            - self
        """
        self.baselineTagger.modelInput(inputData=self.dataTrain, dataType="TRAIN")
        self.baselineTagger.modelTrain()
        templates = [
            Template(Pos([-1])),
            Template(Pos([1])),
            Template(Pos([-2])),
            Template(Pos([2])),
            Template(Pos([-2, -1])),
            Template(Pos([1, 2])),
            Template(Pos([-3, -2, -1])),
            Template(Pos([1, 2, 3])),
            Template(Pos([-1]), Pos([1])),
            Template(Pos([-2]), Pos([-1])),
            Template(Pos([1]), Pos([2])),
            Template(Pos([-1]), Pos([2])),
            Template(Pos([-2]), Pos([1])),
            Template(Word([0])),
            Template(Word([-1])),
            Template(Word([1])),
            Template(Word([-2])),
            Template(Word([2])),
            Template(Word([-2, -1])),
            Template(Word([1, 2])),
            Template(Word([-1, 0])),
            Template(Word([0, 1])),
            Template(Word([-2, 0])),
            Template(Word([0, 2])),
            Template(Word([-1]), Word([1])),
            Template(Word([-2]), Word([-1])),
            Template(Word([1]), Word([2])),
            Template(Word([-1]), Pos([-1])),
            Template(Word([1]), Pos([1])),
            Template(Word([0]), Word([-1]), Pos([-1])),
            Template(Word([0]), Word([1]), Pos([1])),
            Template(Word([1]), Pos([1]), Pos([2])),
            Template(Word([-1]), Pos([-2]), Pos([-1])),
        ]
        # accuracy of in-domain based on templates in library
        # brill24 -> 0.8153575615474795
        # nltkdemo18 -> 0.8007033997655334 
        # nltkdemo18plus -> 0.8007033997655334 
        # fntbl37 -> 0.8156506447831184
        trainer = BrillTaggerTrainer(self.baselineTagger.tagger, templates)
        self.tagger = trainer.train(self.dataTrain)

    def modelTagging(self) -> None:
        """
            This function applies the tagger trained to tag sentences in the testing set, then stores the tagged test
        set as a class parameter;
        :param:
            - self
        """
        self.dataTestTagged = self.tagger.tag_sents([[token[0] for token in sentence] for sentence in self.dataTest])

    def modelPredict(self) -> None:
        """
            This function applies the tagger trained to sentences in the test set, and comparing the tags in prediction
        with the tags as the truth (given with the test set); Printing the accuracy;
        :param:
            - self
        """
        base_accuracy = self.baselineTagger.tagger.accuracy(self.dataTest)
        # print(f"\nThe baseline tagger performs:"
        # f"\n* taggingAccuracy = {base_accuracy}")
        brill_accuracy = self.tagger.accuracy(self.dataTest)
        print(f"\nBased on the input training dataset, the brill performs: "
              f"\n* taggingAccuracy = {brill_accuracy}")

    def modelOutput(self) -> list:
        """
            The output function for the model; This function will return the after-tagging test set (the prediction);
        :param:
            - self,
        :return:
            - self.dataTestTagged, the after-tagging test set as the model prediction;
        """
        return self.dataTestTagged

    def modelAnalyze(self, analyzeErrors=True, analyzeOOVs=True):
        """
            This is the analysis and evaluation method for the model; This function firstly deconstruct the training
        data to creat the vocabulary list, then going through the training and testing sets to analyze and collect the
        errors (differences in prediction=TestTaggedDataset & truth=TestDataset) in tagging and the out-of-vocab cases;
        The analysis report will be printed to the terminal window with all the relevant information about the current
        model; Users may select the option of analyzeErrors=True & analyzeOOVs=True to have mode details;
        :param:
            - self
            - analyzeErrors, boolean, =True for printing more details about errors in tagging (Default: =True);
            - analyzeOOVs, boolean, =True for printing more details about out-of-vocab cases (Default: =True);
        """
        vocab = set([word for sentence in self.dataTrain for (word, tag) in sentence])
        truth = _formatReconstructor(self.dataTest)
        prediction = _formatReconstructor(self.dataTestTagged)

        errorNums = 0
        errors = []
        OOVNums = 0
        OOVs = []
        for couple in zip(truth, prediction):
            if couple[0] != couple[1]:
                errorNums += 1
                errors.append((str(couple[0]).strip('\n'), str(couple[1]).strip('\n')))
            if str(couple[1].split()[0]) not in vocab:
                OOVNums += 1
                OOVs.append((str(couple[0]).strip('\n'), str(couple[1]).strip('\n')))

        print(f"\n--------------------------------------------------------- "
              f"\n* errorNum = {errorNums}"
              f"  --> These words hold unequal PoS tags b/w input TRUTH and model PREDICTION;")
        if analyzeErrors is True:
            print("* with errors in couples:\n")
            for error in errors:
                print(str(error).strip('\n\n'))

        print(f"\n--------------------------------------------------------- "
              f"\n* OOVNums = {OOVNums}"
              f"  --> These words did not appear in the training dataset but appeared in test dataset;")
        if analyzeOOVs is True:
            print("* with OOV in couples:\n")
            for OOV in OOVs:
                print(str(OOV).strip('\n\n'))