import os
import sys
import subprocess
import tempfile
import multiprocessing
import numpy as np 
import scipy as sp 
import pandas as pd
from ConfigParser import SafeConfigParser
from Bio import Seq, SeqIO
import pysam
from bx.bbi.bigwig_file import BigWigFile
from sklearn import linear_model, svm, ensemble, preprocessing, grid_search, metrics

from expt_config_parser import parseExptConfig, parseLibraryConfig

###############################################################################
#                   Import and Merge Training/Test Data                       #
###############################################################################
def loadExperimentData(experimentFile, supportedLibraryPath, library, basePath = '.'):
	libDict, librariesToTables = parseLibraryConfig(os.path.join(supportedLibraryPath, 'library_config.txt'))

	geneTableDict = dict()
	phenotypeTableDict = dict()
	libraryTableDict = dict()

	parser = SafeConfigParser()
	parser.read(experimentFile)
	for exptConfigFile in parser.sections():
		configDict = parseExptConfig(exptConfigFile,libDict)[0]

		libraryTable = pd.read_csv(os.path.join(basePath,configDict['output_folder'],configDict['experiment_name']) + '_librarytable.txt',
			sep='\t', index_col=range(1), header=0)
		libraryTableDict[configDict['experiment_name']] = libraryTable

		geneTable = pd.read_csv(os.path.join(basePath,configDict['output_folder'],configDict['experiment_name']) + '_genetable.txt',
			sep='\t',index_col=range(2),header=range(3))
		phenotypeTable = pd.read_csv(os.path.join(basePath,configDict['output_folder'],configDict['experiment_name']) + '_phenotypetable.txt',\
			sep='\t',index_col=range(1),header=range(2))

		condTups = [(condStr.split(':')[0],condStr.split(':')[1]) for condStr in parser.get(exptConfigFile, 'condition_tuples').strip().split('\n')]
		# print condTups

		geneTableDict[configDict['experiment_name']] = geneTable.loc[:,[level_name for level_name in geneTable.columns if (level_name[0],level_name[1]) in condTups]]
		phenotypeTableDict[configDict['experiment_name']] = phenotypeTable.loc[:,[level_name for level_name in phenotypeTable.columns if (level_name[0],level_name[1]) in condTups]]

	mergedLibraryTable = pd.concat(libraryTableDict.values())
	# print mergedLibraryTable.head()
	mergedLibraryTable_dedup = mergedLibraryTable.drop_duplicates(['gene','sequence'])
	# print mergedLibraryTable_dedup.head()
	mergedGeneTable = pd.concat(geneTableDict.values(), keys=geneTableDict.keys(), axis = 1)
	# print mergedGeneTable.head()
	mergedPhenotypeTable = pd.concat(phenotypeTableDict.values(), keys=phenotypeTableDict.keys(), axis = 1)
	# print mergedPhenotypeTable.head()
	mergedPhenotypeTable_dedup = mergedPhenotypeTable.loc[mergedLibraryTable_dedup.index]

	return mergedLibraryTable_dedup, mergedPhenotypeTable_dedup, mergedGeneTable

def calculateDiscriminantScores(geneTable, effectSize = 'average phenotype of strongest 3', pValue = 'Mann-Whitney p-value'):
	isPseudo = getPseudoIndices(geneTable)
	geneTable_reordered = geneTable.reorder_levels((3,0,1,2), axis=1)
	zscores = geneTable_reordered[effectSize] / geneTable_reordered.loc[isPseudo,effectSize].std()
	pvals = -1 * np.log10(geneTable_reordered[pValue])

	seriesDict = dict()
	for group, table in pd.concat((zscores, pvals), keys=(effectSize,pValue),axis=1).reorder_levels((1,2,3,0), axis=1).groupby(level=range(3),axis=1):
		# print table.head()
		seriesDict[group] = table[group].apply(lambda row: row[effectSize] * row[pValue], axis=1)

	return pd.DataFrame(seriesDict)

def getNormalizedsgRNAsOverThresh(libraryTable, phenotypeTable, discriminantTable, threshold, numToNormalize, transcripts=True):
	maxDiscriminants = pd.concat([discriminantTable.abs().idxmax(axis=1), discriminantTable.abs().max(axis=1)], keys = ('best col','best score'), axis=1)

	if transcripts:
		grouper = (libraryTable['gene'],libraryTable['transcripts'])
	else:
		grouper = libraryTable['gene']

	normedPhenotypes = []
	for name, group in phenotypeTable.groupby(grouper):
		if (transcripts and name[0] == 'negative_control') or (not transcripts and name == 'negative_control'):
			continue
		maxDisc = maxDiscriminants.loc[name]

		if not transcripts:
			maxDisc = maxDisc.sort('best score').iloc[-1]

		if maxDisc['best score'] >= threshold:
			bestGroup = group[maxDisc['best col']]
			normedPhenotypes.append(bestGroup / np.mean(sorted(bestGroup.dropna(), key=abs, reverse=True)[:numToNormalize]))

	return pd.concat(normedPhenotypes), maxDiscriminants

def getGeneFolds(libraryTable, kfold, transcripts=True):
	if transcripts:
		geneGroups = pd.Series(range(len(libraryTable)), index=libraryTable.index).groupby((libraryTable['gene'],libraryTable['transcripts']))
	else:
		geneGroups = pd.Series(range(len(libraryTable)), index=libraryTable.index).groupby(libraryTable['gene'])

	idxList = np.arange(geneGroups.ngroups)
	np.random.shuffle(idxList)

	foldsize = int(np.floor(geneGroups.ngroups * 1.0 / kfold))
	folds = []
	for i in range(kfold):
		testGroups = []
		trainGroups = []
		testSet = set(idxList[i * foldsize: (i+1) * foldsize])
		for i, (name, group) in enumerate(geneGroups):
			if i in testSet:
				testGroups.extend(group.values)
			else:
				trainGroups.extend(group.values)
		folds.append((trainGroups,testGroups))

	return folds


###############################################################################
#                           Calculate sgRNA Parameters                        #
###############################################################################
#tss annotations relying on library input TSSs, may want to convert to gencode in future
def generateTssTable(geneTable, libraryTssFile, cagePeakFile, cageWindow, aliasDict = {'NFIK':'MKI67IP'}):
	codingTssList = []
	with open(libraryTssFile) as infile:
		for line in infile:
			linesplit = line.strip().split('\t')
			try:
				chrom = int(linesplit[2][3:])
			except ValueError:
				chrom = linesplit[2][3:]
			codingTssList.append((chrom, int(linesplit[3]), linesplit[0], linesplit[1], linesplit[2], linesplit[3], linesplit[4]))

	codingTupDict = {(tup[2],tup[3]):tup for tup in codingTssList}

	codingGeneToTransList = dict()

	for geneTrans in codingTupDict:
		if geneTrans[0] not in codingGeneToTransList:
			codingGeneToTransList[geneTrans[0]] = []
			
		codingGeneToTransList[geneTrans[0]].append(geneTrans[1])

	positionList = []
	for (gene,transcriptList), row in geneTable.iterrows():
		
		if gene not in codingGeneToTransList: #only pseudogenes
			positionList.append((np.nan,np.nan,np.nan))
			continue
			
		if transcriptList == 'all':
			positions = [codingTupDict[(gene, trans)][1] for trans in codingGeneToTransList[gene]]
		else:
			positions = [codingTupDict[(gene, trans)][1] for trans in transcriptList.split(',')]
			
		positionList.append((np.mean(positions), codingTupDict[(gene,trans)][6], codingTupDict[(gene,trans)][4]))
			
	tssPositionTable = pd.DataFrame(positionList, index=geneTable.index, columns=['position', 'strand','chromosome'])

	cagePeaks = pysam.Tabixfile(cagePeakFile)
	halfwindow = cageWindow
	strictColor = '60,179,113'
	relaxedColor = '30,144,255'

	cagePeakRanges = []
	for i, (gt, tssRow) in enumerate(tssPositionTable.dropna().iterrows()):
		peaks = cagePeaks.fetch(tssRow['chromosome'],tssRow['position'] - halfwindow,tssRow['position'] + halfwindow, parser=pysam.asBed())
		
		ranges = []
		relaxedRanges = []
		for peak in peaks:
	#         print peak
			if peak.strand == tssRow['strand'] and peak.itemRGB == strictColor:
				ranges.append((peak.start, peak.end))
			elif peak.strand == tssRow['strand'] and peak.itemRGB == relaxedColor:
				relaxedRanges.append((peak.start, peak.end))

		if len(ranges) > 0:
			cagePeakRanges.append(ranges)
		else:
			cagePeakRanges.append(relaxedRanges)
		
	cageSeries = pd.Series(cagePeakRanges, index = tssPositionTable.dropna().index)

	tssPositionTable_cage = pd.concat([tssPositionTable, cageSeries], axis=1)
	tssPositionTable_cage.columns = ['position', 'strand','chromosome','cage peak ranges']
	return tssPositionTable_cage

	# for (gene, transList), row in geneTable.iterrows():
	#   if gene not in gencodeData and gene in aliasDict:
	#       geneData = gencodeData[aliasDict[gene]]
	#   else:
	#       geneData = gencodeData[gene]

def generateTssTable_P1P2strategy(tssTable, cagePeakFile, matchedp1p2Window, anyp1p2Window, anyPeakWindow, minDistanceForTwoTSS, aliasDict):
	cagePeaks = pysam.Tabixfile(cagePeakFile)
	strictColor = '60,179,113'
	relaxedColor = '30,144,255'

	resultRows = []
	for gene, tssRowGroup in tssTable.groupby(level=0):

		if len(set(tssRowGroup['chromosome'].values)) == 1:
			chrom = tssRowGroup['chromosome'].values[0]
		else:
			raise ValueError('mutliple annotated chromosomes for ' + gene)

		if len(set(tssRowGroup['strand'].values)) == 1:
			strand = tssRowGroup['strand'].values[0]
		else:
			raise ValueError('mutliple annotated strands for ' + gene)

		#try to match P1/P2 names within the window
		# peaks = cagePeaks.fetch(chrom,max(0,tssRowGroup['position'].min() - matchedp1p2Window),tssRowGroup['position'].max() + matchedp1p2Window, parser=pysam.asBed())
		peaks = []
		for transcript, row in tssRowGroup.iterrows():
			peaks.extend([p for p in cagePeaks.fetch(chrom,max(0,row['position'] - matchedp1p2Window),row['position'] + matchedp1p2Window, parser=pysam.asBed())])
		p1Matches = set()
		p2Matches = set()
		for peak in peaks:
			if peak.strand == strand and matchPeakName(peak.name, aliasDict[gene] if gene in aliasDict else [gene], 'p1'):
				p1Matches.add((peak.start,peak.end))
			elif peak.strand == strand and matchPeakName(peak.name, aliasDict[gene] if gene in aliasDict else [gene], 'p2') and peak.itemRGB == strictColor:
				p2Matches.add((peak.start,peak.end))
		p1Matches = list(p1Matches)
		p2Matches = list(p2Matches)

		if len(p1Matches) >= 1:
			if len(p1Matches) > 1:
				print 'multiple matched p1:', gene, p1Matches, p2Matches #rare event, typically a doubly-named TSS, basically at the same spot

				closestMatch = p1Matches[0]
				for match in p1Matches:
					if min(abs(match[0] - tssRowGroup['position'])) < min(abs(closestMatch[0] - tssRowGroup['position'])):
						closestMatch = match
				p1Matches = [closestMatch]

			if len(p2Matches) > 1:
				print 'multiple matched p2:', gene, p1Matches, p2Matches

				closestMatch = p2Matches[0]
				for match in p2Matches:
					if min(abs(match[0] - tssRowGroup['position'])) < min(abs(closestMatch[0] - tssRowGroup['position'])):
						closestMatch = match
				p2Matches = [closestMatch]

			if len(p2Matches) == 0 or abs(p1Matches[0][0] - p2Matches[0][0]) <= minDistanceForTwoTSS:
				resultRows.append((gene,'P1P2', chrom, strand, 'CAGE, matched peaks', p1Matches[0], p2Matches[0] if len(p2Matches) > 0 else p1Matches[0]))
			else:
				resultRows.append((gene,'P1', chrom, strand, 'CAGE, matched peaks', p1Matches[0], p1Matches[0]))
				resultRows.append((gene,'P2', chrom, strand, 'CAGE, matched peaks', p2Matches[0], p2Matches[0]))


		#try to match any P1/P2 names 
		else:
			peaks = []
			for transcript, row in tssRowGroup.iterrows():
				peaks.extend([p for p in cagePeaks.fetch(chrom,max(0,row['position'] - anyp1p2Window),row['position'] + anyp1p2Window, parser=pysam.asBed())])
			p1Matches = set()
			p2Matches = set()
			for peak in peaks:
				if peak.strand == strand and peak.name.find('p1@') != -1:
					p1Matches.add((peak.start,peak.end))
				elif peak.strand == strand  and peak.name.find('p2@') != -1 and peak.itemRGB == strictColor:
					p2Matches.add((peak.start,peak.end))
			p1Matches = list(p1Matches)
			p2Matches = list(p2Matches)

			if len(p1Matches) >=1:
				if len(p1Matches) > 1:
					print 'multiple nearby p1:', gene, p1Matches, p2Matches

					closestMatch = p1Matches[0]
					for match in p1Matches:
						if min(abs(match[0] - tssRowGroup['position'])) < min(abs(closestMatch[0] - tssRowGroup['position'])):
							closestMatch = match
					p1Matches = [closestMatch]

				if len(p2Matches) > 1:
					print 'multiple nearby p2:', gene, p1Matches, p2Matches 

					closestMatch = p2Matches[0]
					for match in p2Matches:
						if min(abs(match[0] - tssRowGroup['position'])) < min(abs(closestMatch[0] - tssRowGroup['position'])):
							closestMatch = match
					p2Matches = [closestMatch]

				if len(p2Matches) == 0 or abs(p1Matches[0][0] - p2Matches[0][0]) <= minDistanceForTwoTSS:
					resultRows.append((gene,'P1P2', chrom, strand, 'CAGE, primary peaks', p1Matches[0], p2Matches[0] if len(p2Matches) > 0 else p1Matches[0]))
				else:
					resultRows.append((gene,'P1', chrom, strand, 'CAGE, primary peaks', p1Matches[0], p1Matches[0]))
					resultRows.append((gene,'P2', chrom, strand, 'CAGE, primary peaks', p2Matches[0], p2Matches[0]))


			#try to match robust or permissive peaks
			else:
				for transcript, row in tssRowGroup.iterrows():
					peaks = cagePeaks.fetch(chrom,max(0,row['position']) - anyPeakWindow,row['position'] + anyPeakWindow, parser=pysam.asBed())
					robustPeaks = []
					permissivePeaks = []
					for peak in peaks:
						if peak.strand == strand and peak.itemRGB == strictColor:
							robustPeaks.append((peak.start,peak.end))
						if peak.strand == strand and peak.itemRGB == relaxedColor:
							permissivePeaks.append((peak.start,peak.end))

					if len(robustPeaks) >= 1:
						if strand == '+':
							resultRows.append((gene,transcript[1], chrom, strand, 'CAGE, robust peak', robustPeaks[0], robustPeaks[-1]))
						else:
							resultRows.append((gene,transcript[1], chrom, strand, 'CAGE, robust peak', robustPeaks[-1], robustPeaks[0]))
					elif len(permissivePeaks) >= 1:
						if strand == '+':
							resultRows.append((gene,transcript[1], chrom, strand, 'CAGE permissive peak', permissivePeaks[0], permissivePeaks[-1]))
						else:
							resultRows.append((gene,transcript[1], chrom, strand, 'CAGE permissive peak', permissivePeaks[-1], permissivePeaks[0]))
					else:
						resultRows.append((gene, transcript[1], chrom, strand, 'Annotation', (row['position'],row['position']), (row['position'],row['position'])))

	return pd.DataFrame(resultRows, columns=['gene','transcript','chromosome','strand','TSS source','primary TSS','secondary TSS']).set_index(keys=['gene','transcript'])

def generateSgrnaDistanceTable_p1p2Strategy(sgInfoTable, libraryTable, p1p2Table, transcripts=False):
	sgDistanceSeries = []

	if transcripts == False: # when sgRNAs weren't designed based on the p1p2 strategy
		for name, group in sgInfoTable['pam coordinate'].groupby(libraryTable['gene']):
			if name in p1p2Table.index:
				tssRow = p1p2Table.loc[name]

				if len(tssRow) == 1:
					tssRow = tssRow.iloc[0]
					for sgId, pamCoord in group.iteritems():
						if tssRow['strand'] == '+':
							sgDistanceSeries.append((sgId, name, tssRow.name,
								pamCoord - tssRow['primary TSS'][0],
								pamCoord - tssRow['primary TSS'][1],
								pamCoord - tssRow['secondary TSS'][0],
								pamCoord - tssRow['secondary TSS'][1]))
						else:
							sgDistanceSeries.append((sgId, name, tssRow.name,
								(pamCoord - tssRow['primary TSS'][1]) * -1,
								(pamCoord - tssRow['primary TSS'][0]) * -1,
								(pamCoord - tssRow['secondary TSS'][1]) * -1,
								(pamCoord - tssRow['secondary TSS'][0]) * -1))

				else:
					for sgId, pamCoord in group.iteritems():
						closestTssRow = tssRow.loc[tssRow.apply(lambda row: abs(pamCoord - row['primary TSS'][0]), axis=1).idxmin()]

						if closestTssRow['strand'] == '+':
							sgDistanceSeries.append((sgId, name, closestTssRow.name,
								pamCoord - closestTssRow['primary TSS'][0],
								pamCoord - closestTssRow['primary TSS'][1],
								pamCoord - closestTssRow['secondary TSS'][0],
								pamCoord - closestTssRow['secondary TSS'][1]))
						else:
							sgDistanceSeries.append((sgId, name, closestTssRow.name,
								(pamCoord - closestTssRow['primary TSS'][1]) * -1,
								(pamCoord - closestTssRow['primary TSS'][0]) * -1,
								(pamCoord - closestTssRow['secondary TSS'][1]) * -1,
								(pamCoord - closestTssRow['secondary TSS'][0]) * -1))
	else:
		for name, group in sgInfoTable['pam coordinate'].groupby([libraryTable['gene'],libraryTable['transcripts']]):
			if name in p1p2Table.index:
				tssRow = p1p2Table.loc[[name]]

				if len(tssRow) == 1:
					tssRow = tssRow.iloc[0]
					for sgId, pamCoord in group.iteritems():
						if tssRow['strand'] == '+':
							sgDistanceSeries.append((sgId, tssRow.name[0], tssRow.name[1],
								pamCoord - tssRow['primary TSS'][0],
								pamCoord - tssRow['primary TSS'][1],
								pamCoord - tssRow['secondary TSS'][0],
								pamCoord - tssRow['secondary TSS'][1]))
						else:
							sgDistanceSeries.append((sgId, tssRow.name[0], tssRow.name[1],
								(pamCoord - tssRow['primary TSS'][1]) * -1,
								(pamCoord - tssRow['primary TSS'][0]) * -1,
								(pamCoord - tssRow['secondary TSS'][1]) * -1,
								(pamCoord - tssRow['secondary TSS'][0]) * -1))

				else:
					print name, tssRow
					raise ValueError('all gene/trans pairs should be unique')

	return pd.DataFrame(sgDistanceSeries, columns=['sgId', 'gene', 'transcript', 'primary TSS-Up', 'primary TSS-Down', 'secondary TSS-Up', 'secondary TSS-Down']).set_index(keys=['sgId'])

def generateSgrnaDistanceTable(sgInfoTable, tssTable, libraryTable):
	sgDistanceSeries = []

	for name, group in sgInfoTable['pam coordinate'].groupby([libraryTable['gene'],libraryTable['transcripts']]):
		if name in tssTable.index:
			tssRow = tssTable.loc[name]
			if len(tssRow['cage peak ranges']) != 0:
				spotList = []
				for rangeTup in tssRow['cage peak ranges']:
					spotList.append((rangeTup[0] - tssRow['position']) * (-1 if tssRow['strand'] == '-' else 1))
					spotList.append((rangeTup[1] - tssRow['position']) * (-1 if tssRow['strand'] == '-' else 1))
					
				sgDistanceSeries.append(group.apply(lambda row: distanceMetrics(row, tssRow['position'], min(spotList),max(spotList),tssRow['strand'])))
				
			else:
				sgDistanceSeries.append(group.apply(lambda row: distanceMetrics(row, tssRow['position'], 0, 0, tssRow['strand'])))
		
	return pd.concat(sgDistanceSeries)

def distanceMetrics(position, annotatedTss, cageUp, cageDown, strand):
	relativePos = (position - annotatedTss) * (1 if strand == '+' else -1)
	
	return pd.Series((relativePos, relativePos-cageUp, relativePos-cageDown), index=('annotated','cageUp','cageDown'))

def generateSgrnaLengthSeries(libraryTable):
	lengthSeries =  libraryTable.apply(lambda row: len(row['sequence']),axis=1)
	lengthSeries.name = 'length'
	return lengthSeries

def generateRelativeBasesAndStrand(sgInfoTable, tssTable, libraryTable, genomeDict):
	relbases = []
	strands = []
	sgIds = []
	for gene, sgInfoGroup in sgInfoTable.groupby(libraryTable['gene']):
		tssRowGroup = tssTable.loc[gene]

		if len(set(tssRowGroup['chromosome'].values)) == 1:
			chrom = tssRowGroup['chromosome'].values[0]
		else:
			raise ValueError('mutliple annotated chromosomes for ' + gene)

		if len(set(tssRowGroup['strand'].values)) == 1:
			strand = tssRowGroup['strand'].values[0]
		else:
			raise ValueError('mutliple annotated strands for ' + gene)

		for sg, sgInfo in sgInfoGroup.iterrows():
			sgIds.append(sg)
			geneTup = (sgInfo['gene_name'],','.join(sgInfo['transcript_list']))
			strands.append(True if sgInfo['strand'] == strand else False)

			baseMatrix = []
			for pos in np.arange(-30,10):
				baseMatrix.append(getBaseRelativeToPam(chrom, sgInfo['pam coordinate'],sgInfo['length'], sgInfo['strand'], pos, genomeDict))
			relbases.append(baseMatrix)

	relbases = pd.DataFrame(relbases, index = sgIds, columns = np.arange(-30,10)).loc[libraryTable.index]
	strands = pd.DataFrame(strands, index = sgIds, columns = ['same strand']).loc[libraryTable.index]

	return relbases, strands

def generateBooleanBaseTable(baseTable):
	relbases_bool = []
	for base in ['A','G','C','T']:
		relbases_bool.append(baseTable.applymap(lambda val: val == base))

	return pd.concat(relbases_bool, keys=['A','G','C','T'], axis=1)

def generateBooleanDoubleBaseTable(baseTable):
	doubleBaseTable = []
	tableCols = []
	for b1 in ['A','G','C','T']:
		for b2 in ['A','G','C','T']:
			for i in np.arange(-30,8):
				doubleBaseTable.append(pd.concat((baseTable[i] == b1, baseTable[i+1] == b2),axis=1).all(axis=1))
				tableCols.append(((b1,b2),i))
	return pd.concat(doubleBaseTable, keys=tableCols, axis=1)

def getBaseRelativeToPam(chrom, pamPos, length, strand, relPos, genomeDict):
	rc = {'A':'T','T':'A','G':'C','C':'G','N':'N'}
	#print chrom, pamPos, relPos
	if strand == '+':
		return rc[genomeDict[chrom][pamPos - relPos].upper()]
	elif strand == '-':
		return genomeDict[chrom][pamPos + relPos].upper()
	else:
		raise ValueError()

def getMaxLengthHomopolymer(sequence, base):
	sequence = sequence.upper()
	base = base.upper()
	
	maxBaseCount = 0
	curBaseCount = 0
	for b in sequence:
		if b == base:
			curBaseCount += 1
		else:
			maxBaseCount = max((curBaseCount, maxBaseCount))
			curBaseCount = 0
		
	return max((curBaseCount, maxBaseCount))

def getFractionBaseList(sequence, baseList):
	baseSet = [base.upper() for base in baseList]
	counter = 0.0
	for b in sequence.upper():
		if b in baseSet:
			counter += 1.0
			
	return counter / len(sequence)

#need to fix file naming
def getRNAfoldingTable(libraryTable):
	tempfile_fa = tempfile.NamedTemporaryFile('w+t', delete=False)
	tempfile_rnafold = tempfile.NamedTemporaryFile('w+t', delete=False)

	for name, row in libraryTable.iterrows():
		tempfile_fa.write('>' + name + '\n' + row['sequence'] + '\n')

	tempfile_fa.close()
	tempfile_rnafold.close()
	# print tempfile_fa.name, tempfile_rnafold.name

	subprocess.call('RNAfold --noPS < %s > %s' % (tempfile_fa.name, tempfile_rnafold.name), shell=True)

	mfeSeries_noScaffold = parseViennaMFE(tempfile_rnafold.name, libraryTable)
	isPaired = parseViennaPairing(tempfile_rnafold.name, libraryTable)

	tempfile_fa = tempfile.NamedTemporaryFile('w+t', delete=False)
	tempfile_rnafold = tempfile.NamedTemporaryFile('w+t', delete=False)

	with open(tempfile_fa.name,'w') as outfile:
		for name, row in libraryTable.iterrows():
			outfile.write('>' + name + '\n' + row['sequence'] + 'GTTTAAGAGCTAAGCTGGAAACAGCATAGCAAGTTTAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTTTTT\n')

	tempfile_fa.close()
	tempfile_rnafold.close()
	# print tempfile_fa.name, tempfile_rnafold.name

	subprocess.call('RNAfold --noPS < %s > %s' % (tempfile_fa.name, tempfile_rnafold.name), shell=True)

	mfeSeries_wScaffold = parseViennaMFE(tempfile_rnafold.name, libraryTable)

	return pd.concat((mfeSeries_noScaffold, mfeSeries_wScaffold, isPaired), keys=('no scaffold', 'with scaffold', 'is Paired'), axis=1)

def parseViennaMFE(viennaOutputFile, libraryTable):
	mfeList = []
	with open(viennaOutputFile) as infile:
		for i, line in enumerate(infile):
			if i%3 == 2:
				mfeList.append(float(line.strip().strip('.() ')))
	return pd.Series(mfeList, index=libraryTable.index, name='RNA minimum free energy')

def parseViennaPairing(viennaOutputFile, libraryTable):
	paired = []
	with open(viennaOutputFile) as infile:
		for i, line in enumerate(infile):
			if i%3 == 2:
				foldString = line.strip().split(' ')[0]
				paired.append([char != '.' for char in foldString[-18:]])
	return pd.DataFrame(paired, index=libraryTable.index, columns = range(-20,-2))

def getChromatinDataSeries(bigwigFile, libraryTable, sgInfoTable, tssTable, colname = '', naValue = 0):
	bwindex = BigWigFile(open(bigwigFile))
	chromDict = tssTable['chromosome'].to_dict()

	chromatinScores = []
	for name, sgInfo in sgInfoTable.iterrows():
		geneTup = (sgInfo['gene_name'],','.join(sgInfo['transcript_list']))

		if geneTup not in chromDict: #negative controls
			chromatinScores.append(np.nan)
			continue

		if sgInfo['strand'] == '+':
			sgRange = sgInfo['pam coordinate'] + sgInfo['length']
		else:
			sgRange = sgInfo['pam coordinate'] - sgInfo['length']

		chrom = chromDict[geneTup]
		
		chromatinArray = bwindex.get_as_array(chrom, min(sgInfo['pam coordinate'], sgRange), max(sgInfo['pam coordinate'], sgRange))
		if chromatinArray is not None and len(chromatinArray) > 0:
			chromatinScores.append(np.nanmean(chromatinArray))
		else: #often chrY when using K562 data..
			# print name
			# print chrom, min(sgInfo['pam coordinate'], sgRange), max(sgInfo['pam coordinate'], sgRange)
			chromatinScores.append(np.nan)

	chromatinSeries = pd.Series(chromatinScores, index=libraryTable.index, name = colname)

	return chromatinSeries.fillna(naValue)

def getChromatinDataSeriesByGene(bigwigFileHandle, libraryTable, sgInfoTable, p1p2Table, sgrnaDistanceTable_p1p2, colname = '', naValue = 0, normWindow = 1000):
	bwindex = bigwigFileHandle #BigWigFile(open(bigwigFile))

	chromatinScores = []
	for (gene, transcript), sgInfoGroup in sgInfoTable.groupby([sgrnaDistanceTable_p1p2['gene'], sgrnaDistanceTable_p1p2['transcript']]):   
		tssRow = p1p2Table.loc[[(gene, transcript)]].iloc[0,:]

		chrom = tssRow['chromosome']

		normWindowArray = bwindex.get_as_array(chrom, max(0, tssRow['primary TSS'][0] - normWindow), tssRow['primary TSS'][0] + normWindow)
		if normWindowArray is not None:
			normFactor = np.nanmax(normWindowArray)
		else:
			normFactor = 1

		windowMin = max(0, min(sgInfoGroup['pam coordinate']) - max(sgInfoGroup['length']) - 10)
		windowMax = max(sgInfoGroup['pam coordinate']) + max(sgInfoGroup['length']) + 10
		chromatinWindow = bwindex.get_as_array(chrom, windowMin, windowMax)

		chromatinScores.append(sgInfoGroup.apply(lambda row: getChromatinData(row, chromatinWindow, windowMin, normFactor), axis=1))


	chromatinSeries = pd.concat(chromatinScores)

	return chromatinSeries.fillna(naValue)

def getChromatinData(sgInfoRow, chromatinWindowArray, windowMin, normFactor):
	if sgInfoRow['strand'] == '+':
		sgRange = sgInfoRow['pam coordinate'] + sgInfoRow['length']
	else:
		sgRange = sgInfoRow['pam coordinate'] - sgInfoRow['length']


	if chromatinWindowArray is not None:# and len(chromatinWindowArray) > 0:
		chromatinArray = chromatinWindowArray[min(sgInfoRow['pam coordinate'], sgRange) - windowMin: max(sgInfoRow['pam coordinate'], sgRange) - windowMin]
		return np.nanmean(chromatinArray)/normFactor
	else: #often chrY when using K562 data..
		# print name
		# print chrom, min(sgInfo['pam coordinate'], sgRange), max(sgInfo['pam coordinate'], sgRange)
		return np.nan

def generateTypicalParamTable(libraryTable, sgInfoTable, tssTable, p1p2Table, genomeDict, bwFileHandleDict, transcripts=False):
	lengthSeries = generateSgrnaLengthSeries(libraryTable)

	# sgrnaPositionTable = generateSgrnaDistanceTable(sgInfoTable, tssTable, libraryTable)
	sgrnaPositionTable_p1p2 = generateSgrnaDistanceTable_p1p2Strategy(sgInfoTable, libraryTable, p1p2Table, transcripts)

	baseTable, strand = generateRelativeBasesAndStrand(sgInfoTable, tssTable, libraryTable, genomeDict)
	booleanBaseTable = generateBooleanBaseTable(baseTable)
	doubleBaseTable = generateBooleanDoubleBaseTable(baseTable)

	printNow('.')
	baseList = ['A','G','C','T']
	homopolymerTable = pd.concat([libraryTable.apply(lambda row: np.floor(getMaxLengthHomopolymer(row['sequence'], base)), axis=1) for base in baseList],keys=baseList,axis=1)

	baseFractions = pd.concat([libraryTable.apply(lambda row: getFractionBaseList(row['sequence'], ['A']),axis=1),
							libraryTable.apply(lambda row: getFractionBaseList(row['sequence'], ['G']),axis=1),
							libraryTable.apply(lambda row: getFractionBaseList(row['sequence'], ['C']),axis=1),
							libraryTable.apply(lambda row: getFractionBaseList(row['sequence'], ['T']),axis=1),
							libraryTable.apply(lambda row: getFractionBaseList(row['sequence'], ['G','C']),axis=1),
							libraryTable.apply(lambda row: getFractionBaseList(row['sequence'], ['G','A']),axis=1),
							libraryTable.apply(lambda row: getFractionBaseList(row['sequence'], ['C','A']),axis=1)],keys=['A','G','C','T','GC','purine','CA'],axis=1)

	printNow('.')

	dnaseSeries = getChromatinDataSeriesByGene(bwFileHandleDict['dnase'], libraryTable, sgInfoTable, p1p2Table, sgrnaPositionTable_p1p2)
	printNow('.')
	faireSeries = getChromatinDataSeriesByGene(bwFileHandleDict['faire'], libraryTable, sgInfoTable, p1p2Table, sgrnaPositionTable_p1p2)
	printNow('.')
	mnaseSeries = getChromatinDataSeriesByGene(bwFileHandleDict['mnase'], libraryTable, sgInfoTable, p1p2Table, sgrnaPositionTable_p1p2)
	printNow('.')

	rnafolding = getRNAfoldingTable(libraryTable)

	printNow('Done!')

	return pd.concat([lengthSeries,
		   sgrnaPositionTable_p1p2.iloc[:,2:],
		   homopolymerTable,
		   baseFractions,
		   strand,
		   booleanBaseTable['A'],
		   booleanBaseTable['T'],
		   booleanBaseTable['G'],
		   booleanBaseTable['C'],
		   doubleBaseTable,
		   pd.concat([dnaseSeries,faireSeries,mnaseSeries],keys=['DNase','FAIRE','MNase'], axis=1),
		   rnafolding['no scaffold'],
		   rnafolding['with scaffold'],
		   rnafolding['is Paired']],keys=['length',
		   'distance',
		   'homopolymers',
		   'base fractions',
		   'strand',
		   'base table-A',
		   'base table-T',
		   'base table-G',
		   'base table-C',
		   'base dimers',
		   'accessibility',
		   'RNA folding-no scaffold',
		   'RNA folding-with scaffold',
		   'RNA folding-pairing, no scaffold'],axis=1)

# def generateTypicalParamTable_parallel(libraryTable, sgInfoTable, tssTable, p1p2Table, genomeDict, bwFileHandleDict, processors):
#   processPool = multiprocessing.Pool(processors)

#   colTupList = zip([group for gene, group in libraryTable.groupby(libraryTable['gene'])],
#       [group for gene, group in sgInfoTable.groupby(libraryTable['gene'])])

#   result = processPool.map(lambda colTup: generateTypicalParamTable(colTup[0], colTup[1], tssTable, p1p2Table, genomeDict,bwFileHandleDict), colTupList)

#   return pd.concat(result)

###############################################################################
#                           Learn Parameter Weights                           #
###############################################################################
def fitParams(paramTable, scoreTable, fitTable):
	predictedParams = []
	estimators = []

	for i, (name, col) in enumerate(paramTable.iteritems()):
		
		fitRow = fitTable.iloc[i]

		if fitRow['type'] == 'binary': #binary parameter
			# print name, 'is binary parameter'
			predictedParams.append(col)
			estimators.append('binary')
			
		elif fitRow['type'] == 'continuous':
			col_reshape = col.values.reshape(len(col),1)
			parameters = fitRow['params'] 
			
			svr = svm.SVR(cache_size=500)
			clf = grid_search.GridSearchCV(svr, parameters, n_jobs=16, verbose=0)
			clf.fit(col_reshape, scoreTable)

			print name, clf.best_params_
			predictedParams.append(pd.Series(clf.predict(col_reshape), index=col.index, name=name))
			estimators.append(clf.best_estimator_)

		elif fitRow['type'] == 'binnable':
			parameters = fitRow['params']
			
			assignedBins = binValues(col, parameters['bin width'], parameters['min edge data'])
			groupStats = scoreTable.groupby(assignedBins).agg(parameters['bin function'])
			
			# print name
			# print pd.concat((groupStats,scoreTable.groupby(assignedBins).size()), axis=1)
			
			binnedScores = assignedBins.apply(lambda binVal: groupStats.loc[binVal])
			
			predictedParams.append(binnedScores)
			estimators.append(groupStats)

		elif fitRow['type'] == 'binnable_onehot':
			parameters = fitRow['params']

			assignedBins = binValues(col, parameters['bin width'], parameters['min edge data'])
			binGroups = scoreTable.groupby(assignedBins) 
			groupStats = binGroups.agg(parameters['bin function'])

#             print name
#             print pd.concat((groupStats,scoreTable.groupby(assignedBins).size()), axis=1)
			
			oneHotFrame = pd.DataFrame(np.zeros((len(assignedBins),len(binGroups))), index = assignedBins.index, \
				columns=pd.MultiIndex.from_tuples([(name[0],', '.join([name[1],key])) for key in sorted(binGroups.groups.keys())]))

			for groupName, group in binGroups:
				oneHotFrame.loc[group.index, (name[0],', '.join([name[1],groupName]))] = 1

			predictedParams.append(oneHotFrame)
			estimators.append(groupStats)
		
		else:
			raise ValueError(fitRow['type'] + 'not implemented')

	return pd.concat(predictedParams, axis=1), estimators

def binValues(col, binsize, minEdgePoints=0, edgeOffset = None):
	bins = np.floor(col / binsize) * binsize
	
	if minEdgePoints <= 0:
		if edgeOffset == None:
			return bins.apply(lambda binVal: str(binVal))
		else:
			return bins
	elif minEdgePoints >= len(col):
		raise ValueError('too few data points to meet minimum edge requirements')
	else:
		binGroups = bins.groupby(bins)
		binCounts = binGroups.agg(len).sort_index()
		
		i = 0
		leftBin = []
		if binCounts.iloc[i] < minEdgePoints:
			leftCount = 0
			while leftCount < minEdgePoints:
				leftCount += binCounts.iloc[i]
				leftBin.append(binCounts.index[i])
				i += 1
			
			leftLessThan = binCounts.index[i]
		
		j = -1
		rightBin = []
		if binCounts.iloc[j] < minEdgePoints:
			rightCount = 0
			while rightCount < minEdgePoints:
				rightBin.append(binCounts.index[j])
				rightCount += binCounts.iloc[j]
				j -= 1

			rightMoreThan = binCounts.index[j + 1]
		
		if i > len(binCounts) + j:
			raise ValueError('min edge requirements cannot be met')
		
		if edgeOffset == None: #return strings for bins, fine for grouping, problems for plotting
			return bins.apply(lambda binVal: '< %f' % leftLessThan if binVal in leftBin else('>= %f' % rightMoreThan if binVal in rightBin else str(binVal)))
		else: #apply arbitrary offset instead to ease plotting
			return bins.apply(lambda binVal: leftLessThan - edgeOffset if binVal in leftBin else(rightMoreThan + edgeOffset if binVal in rightBin else binVal))
		
def transformParams(paramTable, fitTable, estimators):
	transformedParams = []
	
	for i, (name, col) in enumerate(paramTable.iteritems()):
		fitRow = fitTable.iloc[i]
		
		if fitRow['type'] == 'binary':
			transformedParams.append(col)
		elif fitRow['type'] == 'continuous':
			col_reshape = col.values.reshape(len(col),1)
			transformedParams.append(pd.Series(estimators[i].predict(col_reshape), index=col.index, name=name))
		elif fitRow['type'] == 'binnable':
			binStats = estimators[i]
			assignedBins = applyBins(col, binStats.index.values)
			transformedParams.append(assignedBins.apply(lambda binVal: binStats.loc[binVal]))

		elif fitRow['type'] == 'binnable_onehot':
			binStats = estimators[i]
			
			assignedBins = applyBins(col, binStats.index.values)
			binGroups = col.groupby(assignedBins)

#             print name
#             print pd.concat((groupStats,scoreTable.groupby(assignedBins).size()), axis=1)
			
			oneHotFrame = pd.DataFrame(np.zeros((len(assignedBins),len(binGroups))), index = assignedBins.index, \
				columns=pd.MultiIndex.from_tuples([(name[0],', '.join([name[1],key])) for key in sorted(binGroups.groups.keys())]))

			for groupName, group in binGroups:
				oneHotFrame.loc[group.index, (name[0],', '.join([name[1],groupName]))] = 1

			transformedParams.append(oneHotFrame)
			
	return pd.concat(transformedParams, axis=1)

def applyBins(column, binStrings):
	leftLabel = ''
	rightLabel = ''
	binTups = []
	for binVal in binStrings:
		if binVal[0] == '<':
			leftLabel = binVal
		elif binVal[0] == '>':
			rightLabel = binVal
			rightBound = float(binVal[3:])
		else:
			binTups.append((float(binVal),binVal))
			
	binTups.sort()
#     print binTups
	leftBound = binTups[0][0]
	if leftLabel == '':
		leftLabel = binTups[0][1]
	
	if rightLabel == '':
		rightLabel = binTups[-1][1]
		rightBound = binTups[-1][0]

	def binFunc(val):
		return leftLabel if val < leftBound else (rightLabel if val >= rightBound else [tup[1] for tup in binTups if val >= tup[0]][-1])

	return column.apply(binFunc)

###############################################################################
#                     Predict sgRNA Scores and Library??                      #
###############################################################################
def findAllGuides(p1p2Table, genomeDict, rangeTup, sgRNALength=20):
	newLibraryTable = []
	newSgInfoTable = []

	for tssTup, tssRow in p1p2Table.iterrows():
		rangeStart = min(min(tssRow['primary TSS']), min(tssRow['secondary TSS'])) + (rangeTup[0] if tssRow['strand'] == '+' else -1 * rangeTup[1])
		rangeEnd = max(max(tssRow['primary TSS']), max(tssRow['secondary TSS'])) + (rangeTup[1] if tssRow['strand'] == '+' else -1 * rangeTup[0])

		genomeRange = str(genomeDict[tssRow['chromosome']][rangeStart:rangeEnd + 1].seq)

		rangeLength = rangeEnd + 1 - rangeStart
		for posOffset in range(rangeLength):
			if genomeRange[posOffset:posOffset+1+1].upper() == 'CC' \
			and posOffset + 3 + sgRNALength < rangeLength \
			and 'N' not in genomeRange[posOffset:posOffset+3+sgRNALength].upper():
				pamCoord = rangeStart+posOffset
				sgId = tssTup[0] + '_' + '+' + '_' + str(pamCoord) + '.' + str(sgRNALength + 3) + '-' + tssTup[1]
				gene = tssTup[0]
				transcripts = tssTup[1]
				rawSequence = genomeRange[posOffset:posOffset+3+sgRNALength]
				sequence = 'G' + str(Seq.Seq(rawSequence[3:-1]).reverse_complement())

				newLibraryTable.append((sgId, gene, transcripts, sequence, rawSequence))
				newSgInfoTable.append((sgId, 'None', gene, sgRNALength + 3, pamCoord, 'not assigned', pamCoord, '+', tssTup[1].split(',')))
			
			elif genomeRange[posOffset-1:posOffset+1].upper() == 'GG' \
			and posOffset - 3 - sgRNALength >= 0 \
			and 'N' not in genomeRange[posOffset + 1 - 3 - sgRNALength:posOffset+1].upper():
				pamCoord = rangeStart+posOffset
				sgId = tssTup[0] + '_' + '-' + '_' + str(pamCoord) + '.' + str(sgRNALength + 3) + '-' + tssTup[1]
				gene = tssTup[0]
				transcripts = tssTup[1]
				rawSequence = genomeRange[posOffset + 1 - 3 - sgRNALength:posOffset+1]
				sequence = 'G' + rawSequence[1:-3]

				newLibraryTable.append((sgId, gene, transcripts, sequence, rawSequence))
				newSgInfoTable.append((sgId, 'None', gene, sgRNALength + 3, pamCoord, 'not assigned', pamCoord, '-', tssTup[1].split(',')))

	return pd.DataFrame(newLibraryTable,columns=['sgId','gene','transcripts','sequence', 'genomic sequence']).set_index('sgId'), \
		pd.DataFrame(newSgInfoTable, columns=['sgId','Sublibrary','gene_name', 'length', 'pam coordinate','pass_score','position','strand', 'transcript_list']).set_index('sgId')

###############################################################################
#                               Utility Functions                             #
###############################################################################
def getPseudoIndices(table):
	return table.apply(lambda row: row.name[0][:6] == 'pseudo', axis=1)

def loadGencodeData(gencodeGTF, indexByENSG = True):
	printNow('Loading annotation file...')
	gencodeData = dict()
	with open(gencodeGTF) as gencodeFile:
		for line in gencodeFile:
			if line[0] != '#':
				linesplit = line.strip().split('\t')
				attrsplit = linesplit[-1].strip('; ').split('; ')
				attrdict = {attr.split(' ')[0]:attr.split(' ')[1].strip('\"') for attr in attrsplit if attr[:3] !='tag'}
				attrdict['tags'] = [attr.split(' ')[1].strip('\"') for attr in attrsplit if attr[:3] == 'tag']

				if indexByENSG:
					dictKey = attrdict['gene_id'].split('.')[0]
				else:
					dictKey = attrdict['gene_name']

				#catch y-linked pseudoautosomal genes
				if 'PAR' in attrdict['tags'] and linesplit[0] == 'chrY':
					continue
				
				if linesplit[2] == 'gene':# and attrdict['gene_type'] == 'protein_coding':
					gencodeData[dictKey] = ([linesplit[0],long(linesplit[3]),long(linesplit[4]),linesplit[6], attrdict],[])
				elif linesplit[2] == 'transcript':
					gencodeData[dictKey][1].append([linesplit[0],long(linesplit[3]),long(linesplit[4]),linesplit[6], attrdict])

	printNow('Done\n')

	return gencodeData

def loadGenomeAsDict(genomeFasta):
	printNow('Loading genome file...')
	genomeDict = SeqIO.to_dict(SeqIO.parse(genomeFasta,'fasta'))
	printNow('Done\n')
	return genomeDict

def loadCageBedData(cageBedFile, matchList = ['p1','p2']):
	cageBedDict = {match:dict() for match in matchList}

	with open(cageBedFile) as infile:
		for line in infile:
			linesplit = line.strip().split('\t')
			
			for name in linesplit[3].split(','):
				namesplit = name.split('@')
				if len(namesplit) == 2:
					for match in matchList:
						if namesplit[0] == match:
							cageBedDict[match][namesplit[1]] = linesplit

	return cageBedDict

def matchPeakName(peakName, geneAliasList, promoterRank):
	for peakString in peakName.split(','):
		peakSplit = peakString.split('@')
		
		if len(peakSplit) == 2\
		and peakSplit[0] == promoterRank\
		and peakSplit[1] in geneAliasList:
			return True
		
		if len(peakSplit) > 2:
			print peakName
		
	return False

def generateAliasDict(hgncFile, gencodeData):
	hgncTable = pd.read_csv(hgncFile,sep='\t', header=0).fillna('')

	geneToAliases = dict()
	geneToENSG = dict()

	for i, row in hgncTable.iterrows():
		geneToAliases[row['Approved Symbol']] = [row['Approved Symbol']]
		geneToAliases[row['Approved Symbol']].extend([] if len(row['Previous Symbols']) == 0 else [name.strip() for name in row['Previous Symbols'].split(',')])
		geneToAliases[row['Approved Symbol']].extend([] if len(row['Synonyms']) == 0 else [name.strip() for name in row['Synonyms'].split(',')])

		geneToENSG[row['Approved Symbol']] = row['Ensembl Gene ID']

	# for gene in gencodeData:
	#   if gene not in geneToAliases:
	#       geneToAliases[gene] = [gene]
		
	#   geneToAliases[gene].extend([tr[-1]['transcript_id'].split('.')[0] for tr in gencodeData[gene][1]])

	return geneToAliases, geneToENSG

#Parse information from the sgRNA ID standard format
def parseSgId(sgId):
    parseDict = dict()
    
    #sublibrary
    if len(sgId.split('=')) == 2:
        parseDict['Sublibrary'] = sgId.split('=')[0]
        remainingId = sgId.split('=')[1]
    else:
        parseDict['Sublibrary'] = None
        remainingId = sgId
        
    #gene name and strand
    underscoreSplit = remainingId.split('_')
    
    for i,item in enumerate(underscoreSplit):
        if item == '+':
            strand = '+'
            geneName = '_'.join(underscoreSplit[:i])
            remainingId = '_'.join(underscoreSplit[i+1:])
            break
        elif item == '-':
            strand = '-'
            geneName = '_'.join(underscoreSplit[:i])
            remainingId = '_'.join(underscoreSplit[i+1:])
            break
        else:
            continue
            
    parseDict['strand'] = strand
    parseDict['gene_name'] = geneName
        
    #position
    dotSplit = remainingId.split('.')
    parseDict['position'] = int(dotSplit[0])
    remainingId = '.'.join(dotSplit[1:])
    
    #length incl pam
    dashSplit = remainingId.split('-')
    parseDict['length'] = int(dashSplit[0])
    remainingId = '-'.join(dashSplit[1:])
    
    #pass score
    tildaSplit = remainingId.split('~')
    parseDict['pass_score'] = tildaSplit[-1]
    remainingId = '~'.join(tildaSplit[:-1]) #should always be length 1 anyway
    
    #transcripts
    parseDict['transcript_list'] = remainingId.split(',')
    
    return parseDict

def parseAllSgIds(libraryTable):
	sgInfoList = []
	for sgId, row in libraryTable.iterrows():
		sgInfo = parseSgId(sgId)

		#fix pam coordinates for -strand ??
		if sgInfo['strand'] == '-':
			sgInfo['pam coordinate'] = sgInfo['position'] #+ 1
		else:
			sgInfo['pam coordinate'] = sgInfo['position']

		sgInfoList.append(sgInfo)

	return pd.DataFrame(sgInfoList, index=libraryTable.index)

def printNow(outputString):
	sys.stdout.write(outputString)
	sys.stdout.flush()
