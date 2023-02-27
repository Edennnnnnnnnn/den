from nltk.tag.brill_trainer import BrillTaggerTrainer
from nltk.tag.brill import Word, Pos
from nltk.tag.hmm import HiddenMarkovModelTrainer, HiddenMarkovModelTagger, _identity, LidstoneProbDist
from nltk.tbl.template import Template


class HMMModel:
	def __init__(self):
		""" Class Variables Initialized """
		# Model Parameter Settings:
		self.tagger = None
	
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
				
	def _modelTrainEstimator(self, fd, bins) -> LidstoneProbDist:
		"""
			The specific estimator defined for HMMModel trainer, be called internally in the training stage of HMMModel;
		:param:
			- self
			- fd, a default parameter for LidstoneProbDist object building;
			- bins, a default parameter for LidstoneProbDist object building;
			"""
		return LidstoneProbDist(fd, 0.025, bins)

	def modelTrain(self) -> None:
		"""
			This is the training control method for the whole process; This function deconstruct the input data and pass
		them into a HMMTrainer, the trainer will then be used to build up a HMMTagger for further operations;
		:param:
			- self
		"""
		tags = [tag for sentence in self.dataTrain for (word, tag) in sentence]
		words = [word for sentence in self.dataTrain for (word, tag) in sentence]
	
		trainer = HiddenMarkovModelTrainer(tags, words)
		HMM = trainer.train_supervised(self.dataTrain, self._modelTrainEstimator)
		HMM = HiddenMarkovModelTagger(
				HMM._symbols,
				HMM._states,
				HMM._transitions,
				HMM._outputs,
				HMM._priors,
				transform=_identity,
		)
		self.tagger = HMM
		
	def modelPredict(self) -> None:
		"""
			This function applies the tagger trained to sentences in the test set, and comparing the tags in prediction
		with the tags as the truth (given with the test set); Printing the accuracy;
		:param:
			- self
		"""
		accuracy = self.tagger.accuracy(self.dataTest)
		print(f"\nBased on the input training dataset, the HMMModel performs: "
				f"\n* taggingAccuracy = {accuracy}")
		
	def modelTagging(self) -> None:
		"""
			This function applies the tagger trained to tag sentences in the testing set, then stores the tagged test
		set as a class parameter;
		:param:
			- self
		"""
		self.dataTestTagged = self.tagger.tag_sents([[token[0] for token in sentence] for sentence in self.dataTest])
		
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
				