
1. Learning sgRNA predictors from empirical data
    * Load scripts and empirical data
    * Generate TSS annotation using FANTOM dataset
    * Calculate parameters for empirical sgRNAs
    * Fit parameters
2. Applying machine learning model to predict sgRNA activity
    * Find all sgRNAs in genomic regions of interest 
    * Predicting sgRNA activity
3. Construct sgRNA libraries
    * Score sgRNAs for off-target potential
* Pick the top sgRNAs for a library, given predicted activity scores and off-target filtering
* Design negative controls matching the base composition of the library
* Finalizing library design

# 1. Learning sgRNA predictors from empirical data
## Load scripts and empirical data


```python
import sys
sys.path.insert(0, '../ScreenProcessing/')
%run sgRNA_learning.py
```


```python
genomeDict = loadGenomeAsDict('large_data_files/hg19.fa')
```

    Loading genome file...Done



```python
#to use pre-calculated sgRNA activity score data (e.g. provided CRISPRi training data), load the following:
libraryTable_training = pd.read_csv('data_files/CRISPRi_trainingdata_libraryTable.txt', sep='\t', index_col = 0)
libraryTable_training.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sublibrary</th>
      <th>gene</th>
      <th>transcripts</th>
      <th>sequence</th>
    </tr>
    <tr>
      <th>sgId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Drug_Targets+Kinase_Phosphatase=AARS_+_70323216.24-all~e39m1</th>
      <td>drug_targets+kinase_phosphatase</td>
      <td>AARS</td>
      <td>all</td>
      <td>GCCCCAGGATCAGGCCCCGCG</td>
    </tr>
    <tr>
      <th>Drug_Targets+Kinase_Phosphatase=AARS_+_70323296.24-all~e39m1</th>
      <td>drug_targets+kinase_phosphatase</td>
      <td>AARS</td>
      <td>all</td>
      <td>GGCCGCCCTCGGAGAGCTCTG</td>
    </tr>
    <tr>
      <th>Drug_Targets+Kinase_Phosphatase=AARS_+_70323318.24-all~e39m1</th>
      <td>drug_targets+kinase_phosphatase</td>
      <td>AARS</td>
      <td>all</td>
      <td>GACGGCGACCCTAGGAGAGGT</td>
    </tr>
    <tr>
      <th>Drug_Targets+Kinase_Phosphatase=AARS_+_70323362.24-all~e39m1</th>
      <td>drug_targets+kinase_phosphatase</td>
      <td>AARS</td>
      <td>all</td>
      <td>GGTGCAGCGGGCCCTTGGCGG</td>
    </tr>
    <tr>
      <th>Drug_Targets+Kinase_Phosphatase=AARS_+_70323441.24-all~e39m1</th>
      <td>drug_targets+kinase_phosphatase</td>
      <td>AARS</td>
      <td>all</td>
      <td>GCGCTCTGATTGGACGGAGCG</td>
    </tr>
  </tbody>
</table>
</div>




```python
sgInfoTable_training = pd.read_csv('data_files/CRISPRi_trainingdata_sgRNAInfoTable.txt', sep='\t', index_col=0)
sgInfoTable_training.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sublibrary</th>
      <th>gene_name</th>
      <th>length</th>
      <th>pam coordinate</th>
      <th>pass_score</th>
      <th>position</th>
      <th>strand</th>
      <th>transcript_list</th>
    </tr>
    <tr>
      <th>sgId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Drug_Targets+Kinase_Phosphatase=AARS_+_70323216.24-all~e39m1</th>
      <td>Drug_Targets+Kinase_Phosphatase</td>
      <td>AARS</td>
      <td>24</td>
      <td>70323216</td>
      <td>e39m1</td>
      <td>70323216</td>
      <td>+</td>
      <td>['all']</td>
    </tr>
    <tr>
      <th>Drug_Targets+Kinase_Phosphatase=AARS_+_70323296.24-all~e39m1</th>
      <td>Drug_Targets+Kinase_Phosphatase</td>
      <td>AARS</td>
      <td>24</td>
      <td>70323296</td>
      <td>e39m1</td>
      <td>70323296</td>
      <td>+</td>
      <td>['all']</td>
    </tr>
    <tr>
      <th>Drug_Targets+Kinase_Phosphatase=AARS_+_70323318.24-all~e39m1</th>
      <td>Drug_Targets+Kinase_Phosphatase</td>
      <td>AARS</td>
      <td>24</td>
      <td>70323318</td>
      <td>e39m1</td>
      <td>70323318</td>
      <td>+</td>
      <td>['all']</td>
    </tr>
    <tr>
      <th>Drug_Targets+Kinase_Phosphatase=AARS_+_70323362.24-all~e39m1</th>
      <td>Drug_Targets+Kinase_Phosphatase</td>
      <td>AARS</td>
      <td>24</td>
      <td>70323362</td>
      <td>e39m1</td>
      <td>70323362</td>
      <td>+</td>
      <td>['all']</td>
    </tr>
    <tr>
      <th>Drug_Targets+Kinase_Phosphatase=AARS_+_70323441.24-all~e39m1</th>
      <td>Drug_Targets+Kinase_Phosphatase</td>
      <td>AARS</td>
      <td>24</td>
      <td>70323441</td>
      <td>e39m1</td>
      <td>70323441</td>
      <td>+</td>
      <td>['all']</td>
    </tr>
  </tbody>
</table>
</div>




```python
activityScores = pd.read_csv('data_files/CRISPRi_trainingdata_activityScores.txt',sep='\t',index_col=0, header=None).iloc[:,0]
activityScores.head()
```




    0
    Drug_Targets+Kinase_Phosphatase=AARS_+_70323216.24-all~e39m1    0.348892
    Drug_Targets+Kinase_Phosphatase=AARS_+_70323296.24-all~e39m1    0.912409
    Drug_Targets+Kinase_Phosphatase=AARS_+_70323318.24-all~e39m1    0.997242
    Drug_Targets+Kinase_Phosphatase=AARS_+_70323362.24-all~e39m1    0.962154
    Drug_Targets+Kinase_Phosphatase=AARS_+_70323441.24-all~e39m1    0.019320
    Name: 1, dtype: float64




```python
tssTable = pd.read_csv('data_files/human_tssTable.txt',sep='\t', index_col=range(2))
tssTable.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>position</th>
      <th>strand</th>
      <th>chromosome</th>
      <th>cage peak ranges</th>
    </tr>
    <tr>
      <th>gene</th>
      <th>transcripts</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A1BG</th>
      <th>all</th>
      <td>58864864</td>
      <td>-</td>
      <td>chr19</td>
      <td>[(58864822, 58864847), (58864848, 58864868)]</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">A1CF</th>
      <th>ENST00000373993.1</th>
      <td>52619744</td>
      <td>-</td>
      <td>chr10</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>ENST00000374001.2,ENST00000373997.3,ENST00000282641.2</th>
      <td>52645434</td>
      <td>-</td>
      <td>chr10</td>
      <td>[(52645379, 52645393), (52645416, 52645444)]</td>
    </tr>
    <tr>
      <th>A2M</th>
      <th>all</th>
      <td>9268752</td>
      <td>-</td>
      <td>chr12</td>
      <td>[(9268547, 9268556), (9268559, 9268568), (9268...</td>
    </tr>
    <tr>
      <th>A2ML1</th>
      <th>all</th>
      <td>8975067</td>
      <td>+</td>
      <td>chr12</td>
      <td>[(8975061, 8975072), (8975101, 8975108), (8975...</td>
    </tr>
  </tbody>
</table>
</div>




```python
p1p2Table = pd.read_csv('data_files/human_p1p2Table.txt',sep='\t', header=0, index_col=range(2))
p1p2Table['primary TSS'] = p1p2Table['primary TSS'].apply(lambda tupString: (int(tupString.strip('()').split(', ')[0].split('.')[0]), int(tupString.strip('()').split(', ')[1].split('.')[0])))
p1p2Table['secondary TSS'] = p1p2Table['secondary TSS'].apply(lambda tupString: (int(tupString.strip('()').split(', ')[0].split('.')[0]),int(tupString.strip('()').split(', ')[1].split('.')[0])))
p1p2Table.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>chromosome</th>
      <th>strand</th>
      <th>TSS source</th>
      <th>primary TSS</th>
      <th>secondary TSS</th>
    </tr>
    <tr>
      <th>gene</th>
      <th>transcript</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">A1BG</th>
      <th>P1</th>
      <td>chr19</td>
      <td>-</td>
      <td>CAGE, matched peaks</td>
      <td>(58858938, 58859039)</td>
      <td>(58858938, 58859039)</td>
    </tr>
    <tr>
      <th>P2</th>
      <td>chr19</td>
      <td>-</td>
      <td>CAGE, matched peaks</td>
      <td>(58864822, 58864847)</td>
      <td>(58864822, 58864847)</td>
    </tr>
    <tr>
      <th>A1CF</th>
      <th>P1P2</th>
      <td>chr10</td>
      <td>-</td>
      <td>CAGE, matched peaks</td>
      <td>(52645379, 52645393)</td>
      <td>(52645379, 52645393)</td>
    </tr>
    <tr>
      <th>A2M</th>
      <th>P1P2</th>
      <td>chr12</td>
      <td>-</td>
      <td>CAGE, matched peaks</td>
      <td>(9268507, 9268523)</td>
      <td>(9268528, 9268542)</td>
    </tr>
    <tr>
      <th>A2ML1</th>
      <th>P1P2</th>
      <td>chr12</td>
      <td>+</td>
      <td>CAGE, matched peaks</td>
      <td>(8975206, 8975223)</td>
      <td>(8975144, 8975169)</td>
    </tr>
  </tbody>
</table>
</div>



## Calculate parameters for empirical sgRNAs

### Because scikit-learn currently does not support any robust method for saving and re-loading the machine learning model, the best strategy is to simply re-learn the model from the training data


```python
#Load bigwig files for any chromatin data of interest
bwhandleDict = {'dnase':BigWigFile(open('large_data_files/wgEncodeOpenChromDnaseK562BaseOverlapSignalV2.bigWig')),
'faire':BigWigFile(open('large_data_files/wgEncodeOpenChromFaireK562Sig.bigWig')),
'mnase':BigWigFile(open('large_data_files/wgEncodeSydhNsomeK562Sig.bigWig'))}
```


```python
paramTable_trainingGuides = generateTypicalParamTable(libraryTable_training,sgInfoTable_training, tssTable, p1p2Table, genomeDict, bwhandleDict)
```

    .....Done!

## Fit parameters


```python
#populate table of fitting parameters
typeList = ['binnable_onehot', 
            'continuous', 'continuous', 'continuous', 'continuous',
            'binnable_onehot','binnable_onehot','binnable_onehot','binnable_onehot',
            'binnable_onehot','binnable_onehot','binnable_onehot','binnable_onehot','binnable_onehot','binnable_onehot','binnable_onehot',
            'binary']
typeList.extend(['binary']*160)
typeList.extend(['binary']*(16*38))
typeList.extend(['binnable_onehot']*3)
typeList.extend(['binnable_onehot']*2)
typeList.extend(['binary']*18)
fitTable = pd.DataFrame(typeList, index=paramTable_trainingGuides.columns, columns=['type'])
fitparams =[{'bin width':1, 'min edge data':50, 'bin function':np.median},
            {'C':[.01,.05, .1,.5], 'gamma':[.000001, .00005,.0001,.0005]},
            {'C':[.01,.05, .1,.5], 'gamma':[.000001, .00005,.0001,.0005]},
            {'C':[.01,.05, .1,.5], 'gamma':[.000001, .00005,.0001,.0005]},
            {'C':[.01,.05, .1,.5], 'gamma':[.000001, .00005,.0001,.0005]},
            {'bin width':1, 'min edge data':50, 'bin function':np.median},
            {'bin width':1, 'min edge data':50, 'bin function':np.median},
            {'bin width':1, 'min edge data':50, 'bin function':np.median},
            {'bin width':1, 'min edge data':50, 'bin function':np.median},
            {'bin width':.1, 'min edge data':50, 'bin function':np.median},
            {'bin width':.1, 'min edge data':50, 'bin function':np.median},
            {'bin width':.1, 'min edge data':50, 'bin function':np.median},
            {'bin width':.1, 'min edge data':50, 'bin function':np.median},
            {'bin width':.1, 'min edge data':50, 'bin function':np.median},
            {'bin width':.1, 'min edge data':50, 'bin function':np.median},
            {'bin width':.1, 'min edge data':50, 'bin function':np.median},dict()]
fitparams.extend([dict()]*160)
fitparams.extend([dict()]*(16*38))
fitparams.extend([
            {'bin width':.15, 'min edge data':50, 'bin function':np.median},
            {'bin width':.15, 'min edge data':50, 'bin function':np.median},
            {'bin width':.15, 'min edge data':50, 'bin function':np.median}])
fitparams.extend([
            {'bin width':2, 'min edge data':50, 'bin function':np.median},
            {'bin width':2, 'min edge data':50, 'bin function':np.median}])
fitparams.extend([dict()]*18)
fitTable['params'] = fitparams
```


```python
#load in the 5-fold cross-validation splits used to generate the model
import cPickle
with open('data_files/CRISPRi_trainingdata_traintestsets.txt') as infile:
    geneFold_train, geneFold_test = cPickle.load(infile)
```


```python
transformedParams_train, estimators = fitParams(paramTable_trainingGuides.loc[activityScores.dropna().index].iloc[geneFold_train], activityScores.loc[activityScores.dropna().index].iloc[geneFold_train], fitTable)

transformedParams_test = transformParams(paramTable_trainingGuides.loc[activityScores.dropna().index].iloc[geneFold_test], fitTable, estimators)

reg = linear_model.ElasticNetCV(l1_ratio=[.5, .75, .9, .99,1], n_jobs=16, max_iter=2000)

scaler = preprocessing.StandardScaler()
reg.fit(scaler.fit_transform(transformedParams_train), activityScores.loc[activityScores.dropna().index].iloc[geneFold_train])
predictedScores = pd.Series(reg.predict(scaler.transform(transformedParams_test)), index=transformedParams_test.index)
testScores = activityScores.loc[activityScores.dropna().index].iloc[geneFold_test]

print 'Prediction AUC-ROC:', metrics.roc_auc_score((testScores >= .75).values, np.array(predictedScores.values,dtype='float64'))
print 'Prediction R^2:', reg.score(scaler.transform(transformedParams_test), testScores)
print 'Regression parameters:', reg.l1_ratio_, reg.alpha_
coefs = pd.DataFrame(zip(*[abs(reg.coef_),reg.coef_]), index = transformedParams_test.columns, columns=['abs','true'])
print 'Number of features used:', len(coefs) - sum(coefs['abs'] < .00000000001)
```

    ('distance', 'primary TSS-Up') {'C': 0.05, 'gamma': 0.0001}
    ('distance', 'primary TSS-Down') {'C': 0.5, 'gamma': 5e-05}
    ('distance', 'secondary TSS-Up') {'C': 0.1, 'gamma': 5e-05}
    ('distance', 'secondary TSS-Down') {'C': 0.1, 'gamma': 5e-05}
    Prediction AUC-ROC: 0.803109696478
    Prediction R^2: 0.31263687609
    Regression parameters: 0.5 0.00534455043278



    

    NameErrorTraceback (most recent call last)

    <ipython-input-17-c622a0e7d525> in <module>()
         13 print 'Prediction R^2:', reg.score(scaler.transform(transformedParams_test), testScores)
         14 print 'Regression parameters:', reg.l1_ratio_, reg.alpha_
    ---> 15 coefs.append(pd.DataFrame(zip(*[abs(reg.coef_),reg.coef_]), index = transformedParams_test.columns, columns=['abs','true']))
         16 print 'Number of features used:', len(coefs[-1]) - sum(coefs[-1]['abs'] < .00000000001)


    NameError: name 'coefs' is not defined



```python
#can save state for reproducing estimators later
#the pickling of the scikit-learn estimators/regressors will allow the model to be reloaded for prediction of other guide designs, 
#   but will not be compatible across scikit-learn versions, so it is important to preserve the training data and training/test folds
import cPickle
estimatorString = cPickle.dumps((fitTable, estimators, scaler, reg))
with open(PICKLE_FILE,'w') as outfile:
    outfile.write(estimatorString)
    
#also save the transformed parameters as these can slightly differ based on the automated binning strategy
transformedParams_train.head().to_csv(TRANSFORMED_PARAM_HEADER,sep='\t')
```

# 2. Applying machine learning model to predict sgRNA activity

## Generate TSS annotation using FANTOM dataset


```python
#you can supply any table of gene transcription start sites formatted as below
#for demonstration purposes, the rest of this walkthrough will use a small arbitrary subset of the protein coding TSS table
tssTable_new = tssTable.iloc[10:20, :-1]
tssTable_new.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>position</th>
      <th>strand</th>
      <th>chromosome</th>
    </tr>
    <tr>
      <th>gene</th>
      <th>transcripts</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AADACL2</th>
      <th>all</th>
      <td>151451714</td>
      <td>+</td>
      <td>chr3</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">AADAT</th>
      <th>ENST00000337664.4</th>
      <td>171011117</td>
      <td>-</td>
      <td>chr4</td>
    </tr>
    <tr>
      <th>ENST00000337664.4,ENST00000509167.1,ENST00000353187.2</th>
      <td>171011284</td>
      <td>-</td>
      <td>chr4</td>
    </tr>
    <tr>
      <th>ENST00000509167.1,ENST00000515480.1,ENST00000353187.2</th>
      <td>171011424</td>
      <td>-</td>
      <td>chr4</td>
    </tr>
    <tr>
      <th>AAED1</th>
      <th>all</th>
      <td>99417562</td>
      <td>-</td>
      <td>chr9</td>
    </tr>
  </tbody>
</table>
</div>




```python
#if desired, use the ensembl annotation and the HGNC database to supply gene aliases to assist P1P2 matching in the next step
gencodeData = loadGencodeData('large_data_files/gencode.v19.annotation.gtf')
geneToAliases = generateAliasDict('large_data_files/20150424_HGNC_symbols.txt',gencodeData)
```

    Loading annotation file...Done



```python
#Now create a TSS annotation by searching for P1 and P2 peaks near annotated TSSs
#same parameters as for our lncRNA libraries
p1p2Table_new = generateTssTable_P1P2strategy(tssTable_new, 'large_data_files/TSS_human.sorted.bed.gz', 
                                                  matchedp1p2Window = 30000, #region around supplied TSS annotation to search for a FANTOM P1 or P2 peak that matches the gene name (or alias)
                                                  anyp1p2Window = 500, #region around supplied TSS annotation to search for the nearest P1 or P2 peak
                                                  anyPeakWindow = 200, #region around supplied TSS annotation to search for any CAGE peak
                                                  minDistanceForTwoTSS = 1000, #If a P1 and P2 peak are found, maximum distance at which to combine into a single annotation (with primary/secondary TSS positions)
                                                  aliasDict = geneToAliases[0])
#the function will report some collisions of IDs due to use of aliases and redundancy in genome, but will resolve these itself
```


```python
p1p2Table_new.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>chromosome</th>
      <th>strand</th>
      <th>TSS source</th>
      <th>primary TSS</th>
      <th>secondary TSS</th>
    </tr>
    <tr>
      <th>gene</th>
      <th>transcript</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AADACL2</th>
      <th>P1P2</th>
      <td>chr3</td>
      <td>+</td>
      <td>CAGE, matched peaks</td>
      <td>(151451707, 151451722)</td>
      <td>(151451707, 151451722)</td>
    </tr>
    <tr>
      <th>AADAT</th>
      <th>P1P2</th>
      <td>chr4</td>
      <td>-</td>
      <td>CAGE, matched peaks</td>
      <td>(171011323, 171011408)</td>
      <td>(171011084, 171011147)</td>
    </tr>
    <tr>
      <th>AAED1</th>
      <th>P1P2</th>
      <td>chr9</td>
      <td>-</td>
      <td>CAGE, matched peaks</td>
      <td>(99417562, 99417609)</td>
      <td>(99417615, 99417622)</td>
    </tr>
    <tr>
      <th>AAGAB</th>
      <th>P1P2</th>
      <td>chr15</td>
      <td>-</td>
      <td>CAGE, matched peaks</td>
      <td>(67546963, 67547024)</td>
      <td>(67546963, 67547024)</td>
    </tr>
    <tr>
      <th>AAK1</th>
      <th>P1P2</th>
      <td>chr2</td>
      <td>-</td>
      <td>CAGE, matched peaks</td>
      <td>(69870747, 69870812)</td>
      <td>(69870854, 69870878)</td>
    </tr>
  </tbody>
</table>
</div>




```python
p1p2Table_new.groupby('TSS source').agg(len).iloc[:,[2]]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>primary TSS</th>
    </tr>
    <tr>
      <th>TSS source</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CAGE, matched peaks</th>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(p1p2Table_new)
```




    8




```python
#save tables
tssTable_alllncs.to_csv(TSS_TABLE_PATH,sep='\t', index_col=range(2))
p1p2Table_alllncs.to_csv(P1P2_TABLE_PATH,sep='\t', header=0, index_col=range(2))
```

## Find all sgRNAs in genomic regions of interest 


```python
libraryTable_new, sgInfoTable_new = findAllGuides(p1p2Table_new, genomeDict, 
                                                  (-25,500)) #region around P1P2 TSSs to search for new sgRNAs; recommend -550,-25 for CRISPRa
```


```python
len(libraryTable_new)
```




    1125




```python
libraryTable_new.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gene</th>
      <th>transcripts</th>
      <th>sequence</th>
      <th>genomic sequence</th>
    </tr>
    <tr>
      <th>sgId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AADACL2_+_151451720.23-P1P2</th>
      <td>AADACL2</td>
      <td>P1P2</td>
      <td>GTAGACTTGGGAACTCTCTC</td>
      <td>CCTGAGAGAGTTCCCAAGTCTAC</td>
    </tr>
    <tr>
      <th>AADACL2_+_151451732.23-P1P2</th>
      <td>AADACL2</td>
      <td>P1P2</td>
      <td>GGTAGAGCAATTGTAGACTT</td>
      <td>CCCAAGTCTACAATTGCTCTACT</td>
    </tr>
    <tr>
      <th>AADACL2_+_151451733.23-P1P2</th>
      <td>AADACL2</td>
      <td>P1P2</td>
      <td>GAGTAGAGCAATTGTAGACT</td>
      <td>CCAAGTCTACAATTGCTCTACTA</td>
    </tr>
    <tr>
      <th>AADACL2_-_151451809.23-P1P2</th>
      <td>AADACL2</td>
      <td>P1P2</td>
      <td>GCTCAGTACTGTGAAGAAGC</td>
      <td>TCTCAGTACTGTGAAGAAGCTGG</td>
    </tr>
    <tr>
      <th>AADACL2_-_151451816.23-P1P2</th>
      <td>AADACL2</td>
      <td>P1P2</td>
      <td>GCTGTGAAGAAGCTGGAAAA</td>
      <td>ACTGTGAAGAAGCTGGAAAAAGG</td>
    </tr>
  </tbody>
</table>
</div>



## Predicting sgRNA activity


```python
#calculate parameters for new sgRNAs
paramTable_new = generateTypicalParamTable(libraryTable_new, sgInfoTable_new, tssTable_new, p1p2Table_new, genomeDict, bwhandleDict)
#this ran, but notebook disconnected so wasn't marked as run
```

    .....Done!


```python
paramTable_new.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>length</th>
      <th colspan="4" halign="left">distance</th>
      <th colspan="4" halign="left">homopolymers</th>
      <th>base fractions</th>
      <th>...</th>
      <th colspan="10" halign="left">RNA folding-pairing, no scaffold</th>
    </tr>
    <tr>
      <th></th>
      <th>length</th>
      <th>primary TSS-Up</th>
      <th>primary TSS-Down</th>
      <th>secondary TSS-Up</th>
      <th>secondary TSS-Down</th>
      <th>A</th>
      <th>G</th>
      <th>C</th>
      <th>T</th>
      <th>A</th>
      <th>...</th>
      <th>-12</th>
      <th>-11</th>
      <th>-10</th>
      <th>-9</th>
      <th>-8</th>
      <th>-7</th>
      <th>-6</th>
      <th>-5</th>
      <th>-4</th>
      <th>-3</th>
    </tr>
    <tr>
      <th>sgId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AADACL2_+_151451720.23-P1P2</th>
      <td>20</td>
      <td>13</td>
      <td>-2</td>
      <td>13</td>
      <td>-2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0.20</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>AADACL2_+_151451732.23-P1P2</th>
      <td>20</td>
      <td>25</td>
      <td>10</td>
      <td>25</td>
      <td>10</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0.30</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>AADACL2_+_151451733.23-P1P2</th>
      <td>20</td>
      <td>26</td>
      <td>11</td>
      <td>26</td>
      <td>11</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0.35</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>AADACL2_-_151451809.23-P1P2</th>
      <td>20</td>
      <td>102</td>
      <td>87</td>
      <td>102</td>
      <td>87</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.30</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>AADACL2_-_151451816.23-P1P2</th>
      <td>20</td>
      <td>109</td>
      <td>94</td>
      <td>109</td>
      <td>94</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0.40</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 808 columns</p>
</div>




```python
#if starting from a separate session from where you ran the sgRNA learning steps of Part 1, reload the following
import cPickle
with open(PICKLE_FILE) as infile:
    fitTable, estimators, scaler, reg = cPickle.load(infile)
    
transformedParams_train = pd.read_csv(TRANSFORMED_PARAM_HEADER,sep='\t')
```


```python
#transform and predict scores according to sgRNA prediction model
transformedParams_new = transformParams(paramTable_new, fitTable, estimators)

#reconcile any differences in column headers generated by automated binning
colTups = []
for (l1, l2), col in transformedParams_new.iteritems():
    colTups.append((l1,str(l2)))
transformedParams_new.columns = pd.MultiIndex.from_tuples(colTups)

predictedScores_new = pd.Series(reg.predict(scaler.transform(transformedParams_new.loc[:, transformedParams_train.columns].fillna(0).values)), index=transformedParams_new.index)
```


```python
predictedScores_new.head()
```




    sgId
    AADACL2_+_151451720.23-P1P2    0.641245
    AADACL2_+_151451732.23-P1P2    0.693926
    AADACL2_+_151451733.23-P1P2    0.655759
    AADACL2_-_151451809.23-P1P2    0.500835
    AADACL2_-_151451816.23-P1P2    0.434376
    dtype: float64




```python
libraryTable_new.to_csv(LIBRARY_TABLE_PATH,sep='\t')
sgInfoTable_new.to_csv(sgRNA_INFO_PATH,sep='\t')
predictedScores_new.to_csv(PREDICTED_SCORES_PATH, sep='\t')
```

# 3. Construct sgRNA libraries
## Score sgRNAs for off-target potential


```python
#There are many ways to score sgRNAs as off-target; below is one listed one method that is simple and flexible,
#but ignores gapped alignments, alternate PAMs, and uses bowtie which may not be maximally sensitive in all cases
```


```python
#output all sequences to a temporary FASTQ file for running bowtie alignment
fqFile = 'temp_bowtie.fq'

def outputTempBowtieFastq(libraryTable, outputFileName):
    phredString = 'I4!=======44444+++++++' #weighting for how impactful mismatches are along sgRNA sequence 
    with open(outputFileName,'w') as outfile:
        for name, row in libraryTable.iterrows():
            outfile.write('@' + name + '\n')
            outfile.write('CCN' + str(Seq.Seq(row['sequence'][1:]).reverse_complement()) + '\n')
            outfile.write('+\n')
            outfile.write(phredString + '\n')
            
outputTempBowtieFastq(libraryTable_new, fqFile)
```


```python
import subprocess

#specifying a list of parameters to run bowtie with
#each tuple contains
# *the mismatch threshold below which a site is considered a potential off-target (higher is more stringent)
# *the number of sites allowed (1 is minimum since each sgRNA should have one true site in genome)
# *the genome index against which to align the sgRNA sequences; these can be custom built to only consider sites near TSSs
# *a name for the bowtie run to create appropriately named output files
alignmentList = [(39,1,'large_data_files/hg19.ensemblTSSflank500b','39_nearTSS'),
                (31,1,'large_data_files/hg19.ensemblTSSflank500b','31_nearTSS'),
                (21,1,'large_data_files/hg19_maskChrMandPAR','21_genome'),
                (31,2,'large_data_files/hg19.ensemblTSSflank500b','31_2_nearTSS'),
                (31,3,'large_data_files/hg19.ensemblTSSflank500b','31_3_nearTSS')]

alignmentColumns = []
for btThreshold, mflag, bowtieIndex, runname in alignmentList:

    alignedFile = 'bowtie_output/' + runname + '_aligned.txt'
    unalignedFile = 'bowtie_output/' + runname + '_unaligned.fq'
    maxFile = 'bowtie_output/' + runname + '_max.fq'
    
    bowtieString = 'bowtie -n 3 -l 15 -e '+str(btThreshold)+' -m ' + str(mflag) + ' --nomaqround -a --tryhard -p 16 --chunkmbs 256 ' + bowtieIndex + ' --suppress 5,6,7 --un ' + unalignedFile + ' --max ' + maxFile + ' '+ ' -q '+fqFile+' '+ alignedFile
    print bowtieString
    print subprocess.call(bowtieString, shell=True) #0 means finished without errors

    #parse through the file of sgRNAs that exceeded "m", the maximum allowable alignments, and mark "True" any that are found
    try:
        with open(maxFile) as infile:
            sgsAligning = set()
            for i, line in enumerate(infile):
                if i%4 == 0: #id line
                    sgsAligning.add(line.strip()[1:])
    except IOError: #no sgRNAs exceeded m, so no maxFile created
        sgsAligning = set()
                    
    alignmentColumns.append(libraryTable_new.apply(lambda row: row.name in sgsAligning, axis=1))
    
#collate results into a table, and flip the boolean values to yield the sgRNAs that passed filter as True
alignmentTable = pd.concat(alignmentColumns,axis=1, keys=zip(*alignmentList)[3]).ne(True)
```

    bowtie -n 3 -l 15 -e 39 -m 1 --nomaqround -a --tryhard -p 16 --chunkmbs 256 large_data_files/hg19.ensemblTSSflank500b --suppress 5,6,7 --un bowtie_output/39_nearTSS_unaligned.fq --max bowtie_output/39_nearTSS_max.fq  -q temp_bowtie.fq bowtie_output/39_nearTSS_aligned.txt
    0
    bowtie -n 3 -l 15 -e 31 -m 1 --nomaqround -a --tryhard -p 16 --chunkmbs 256 large_data_files/hg19.ensemblTSSflank500b --suppress 5,6,7 --un bowtie_output/31_nearTSS_unaligned.fq --max bowtie_output/31_nearTSS_max.fq  -q temp_bowtie.fq bowtie_output/31_nearTSS_aligned.txt
    0
    bowtie -n 3 -l 15 -e 21 -m 1 --nomaqround -a --tryhard -p 16 --chunkmbs 256 large_data_files/hg19_maskChrMandPAR --suppress 5,6,7 --un bowtie_output/21_genome_unaligned.fq --max bowtie_output/21_genome_max.fq  -q temp_bowtie.fq bowtie_output/21_genome_aligned.txt
    0
    bowtie -n 3 -l 15 -e 31 -m 2 --nomaqround -a --tryhard -p 16 --chunkmbs 256 large_data_files/hg19.ensemblTSSflank500b --suppress 5,6,7 --un bowtie_output/31_2_nearTSS_unaligned.fq --max bowtie_output/31_2_nearTSS_max.fq  -q temp_bowtie.fq bowtie_output/31_2_nearTSS_aligned.txt
    0
    bowtie -n 3 -l 15 -e 31 -m 3 --nomaqround -a --tryhard -p 16 --chunkmbs 256 large_data_files/hg19.ensemblTSSflank500b --suppress 5,6,7 --un bowtie_output/31_3_nearTSS_unaligned.fq --max bowtie_output/31_3_nearTSS_max.fq  -q temp_bowtie.fq bowtie_output/31_3_nearTSS_aligned.txt
    0



```python
alignmentTable.head() #True = passed threshold
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>39_nearTSS</th>
      <th>31_nearTSS</th>
      <th>21_genome</th>
      <th>31_2_nearTSS</th>
      <th>31_3_nearTSS</th>
    </tr>
    <tr>
      <th>sgId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AADACL2_+_151451720.23-P1P2</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>AADACL2_+_151451732.23-P1P2</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>AADACL2_+_151451733.23-P1P2</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>AADACL2_-_151451809.23-P1P2</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>AADACL2_-_151451816.23-P1P2</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## Pick the top sgRNAs for a library, given predicted activity scores and off-target filtering


```python
#combine all generated data into one master table
predictedScores_new.name = 'predicted score'
v2Table = pd.concat((libraryTable_new, predictedScores_new, alignmentTable, sgInfoTable_new), axis=1, keys=['library table v2', 'predicted score', 'off-target filters', 'sgRNA info'])
```


```python
import re
#for our pCRISPRi/a-v2 vector, we append flanking sequences to each sgRNA sequence for cloning and require the oligo to contain
#exactly 1 BstXI and BlpI site each for cloning, and exactly 0 SbfI sites for sequencing sample preparation
restrictionSites = {re.compile('CCA......TGG'):1,
                   re.compile('GCT.AGC'):1,
                   re.compile('CCTGCAGG'):0}

def matchREsites(sequence, REdict):
    seq = sequence.upper()
    for resite, numMatchesExpected in restrictionSites.iteritems():
        if len(resite.findall(seq)) != numMatchesExpected:
            return False
        
    return True

def checkOverlaps(leftPosition, acceptedLeftPositions, nonoverlapMin):
    for pos in acceptedLeftPositions:
        if abs(pos - leftPosition) < nonoverlapMin:
            return False
    return True
```


```python
#flanking sequences
upstreamConstant = 'CCACCTTGTTG'
downstreamConstant = 'GTTTAAGAGCTAAGCTG'

#minimum overlap between two sgRNAs targeting the same TSS
nonoverlapMin = 3

#number of sgRNAs to pick per gene/TSS
sgRNAsToPick = 10

#list of off-target filter (or combinations of filters) levels, matching the names in the alignment table above
offTargetLevels = [['31_nearTSS', '21_genome'],
                  ['31_nearTSS'],
                  ['21_genome'],
                  ['31_2_nearTSS'],
                  ['31_3_nearTSS']]

#for each gene/TSS, go through each sgRNA in descending order of predicted score
#if an sgRNA passes the restriction site, overlap, and off-target filters, accept it into the library
#if the number of sgRNAs accepted is less than sgRNAsToPick, reduce off-target stringency by one and continue
v2Groups = v2Table.groupby([('library table v2','gene'),('library table v2','transcripts')])
newSgIds = []
unfinishedTss = []
for (gene, transcript), group in v2Groups:
    geneSgIds = []
    geneLeftPositions = []
    empiricalSgIds = dict()
    
    stringency = 0
    
    while len(geneSgIds) < sgRNAsToPick and stringency < len(offTargetLevels):
        for sgId_v2, row in group.sort_values(('predicted score','predicted score'), ascending=False).iterrows():
            oligoSeq = upstreamConstant + row[('library table v2','sequence')] + downstreamConstant
            leftPos = row[('sgRNA info', 'position')] - (23 if row[('sgRNA info', 'strand')] == '-' else 0)
            if len(geneSgIds) < sgRNAsToPick and row['off-target filters'].loc[offTargetLevels[stringency]].all() \
                and matchREsites(oligoSeq, restrictionSites) \
                and checkOverlaps(leftPos, geneLeftPositions, nonoverlapMin):
                geneSgIds.append((sgId_v2,
                                  gene,transcript,
                                  row[('library table v2','sequence')], oligoSeq,
                                  row[('predicted score','predicted score')], np.nan,
                                 stringency))
                geneLeftPositions.append(leftPos)
                
        stringency += 1
            
    if len(geneSgIds) < sgRNAsToPick:
        unfinishedTss.append((gene, transcript)) #if the number of accepted sgRNAs is still less than sgRNAsToPick, discard gene
    else:
        newSgIds.extend(geneSgIds)
        
libraryTable_complete = pd.DataFrame(newSgIds, columns = ['sgID', 'gene', 'transcript','protospacer sequence', 'oligo sequence',
 'predicted score', 'empirical score', 'off-target stringency']).set_index('sgID')
```


```python
print len(libraryTable_complete)
```

    80



```python
#number of sgRNAs accepted at each stringency level
libraryTable_complete.groupby('off-target stringency').agg(len).iloc[:,0]
```




    off-target stringency
    0    80
    Name: gene, dtype: int64




```python
#number of TSSs with fewer than required number of sgRNAs (and thus not included in the library)
print len(unfinishedTss)
```

    0



```python
libraryTable_complete.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gene</th>
      <th>transcript</th>
      <th>protospacer sequence</th>
      <th>oligo sequence</th>
      <th>predicted score</th>
      <th>empirical score</th>
      <th>off-target stringency</th>
    </tr>
    <tr>
      <th>sgID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AADACL2_+_151451732.23-P1P2</th>
      <td>AADACL2</td>
      <td>P1P2</td>
      <td>GGTAGAGCAATTGTAGACTT</td>
      <td>CCACCTTGTTGGGTAGAGCAATTGTAGACTTGTTTAAGAGCTAAGCTG</td>
      <td>0.693926</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AADACL2_-_151452019.23-P1P2</th>
      <td>AADACL2</td>
      <td>P1P2</td>
      <td>GATGACTTATTGACTAAAAA</td>
      <td>CCACCTTGTTGGATGACTTATTGACTAAAAAGTTTAAGAGCTAAGCTG</td>
      <td>0.451392</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AADACL2_+_151452121.23-P1P2</th>
      <td>AADACL2</td>
      <td>P1P2</td>
      <td>GACTGTTACTCACAGATATA</td>
      <td>CCACCTTGTTGGACTGTTACTCACAGATATAGTTTAAGAGCTAAGCTG</td>
      <td>0.426695</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AADACL2_-_151451828.23-P1P2</th>
      <td>AADACL2</td>
      <td>P1P2</td>
      <td>GTGGAAAAAGGGATATTATG</td>
      <td>CCACCTTGTTGGTGGAAAAAGGGATATTATGGTTTAAGAGCTAAGCTG</td>
      <td>0.404655</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AADACL2_-_151451931.23-P1P2</th>
      <td>AADACL2</td>
      <td>P1P2</td>
      <td>GAGCTGGAAAATAATGGCCT</td>
      <td>CCACCTTGTTGGAGCTGGAAAATAATGGCCTGTTTAAGAGCTAAGCTG</td>
      <td>0.404269</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Design negative controls matching the base composition of the library


```python
#calcluate the base frequency at each position of the sgRNA, then generate random sequences weighted by this frequency
def getBaseFrequencies(libraryTable, baseConversion = {'G':0, 'C':1, 'T':2, 'A':3}):
    baseArray = np.zeros((len(libraryTable),20))

    for i, (index, seq) in enumerate(libraryTable['protospacer sequence'].iteritems()):
        for j, char in enumerate(seq.upper()):
            baseArray[i,j] = baseConversion[char]

    baseTable = pd.DataFrame(baseArray, index = libraryTable.index)
    
    baseFrequencies = baseTable.apply(lambda col: col.groupby(col).agg(len)).fillna(0) / len(baseTable)
    baseFrequencies.index = ['G','C','T','A']
    
    baseCumulativeFrequencies = baseFrequencies.copy()
    baseCumulativeFrequencies.loc['C'] = baseFrequencies.loc['G'] + baseFrequencies.loc['C']
    baseCumulativeFrequencies.loc['T'] = baseFrequencies.loc['G'] + baseFrequencies.loc['C'] + baseFrequencies.loc['T']
    baseCumulativeFrequencies.loc['A'] = baseFrequencies.loc['G'] + baseFrequencies.loc['C'] + baseFrequencies.loc['T'] + baseFrequencies.loc['A']

    return baseFrequencies, baseCumulativeFrequencies

def generateRandomSequence(baseCumulativeFrequencies):
    randArray = np.random.random(baseCumulativeFrequencies.shape[1])
    
    seq = []
    for i, col in baseCumulativeFrequencies.iteritems():
        for base, freq in col.iteritems():
            if randArray[i] < freq:
                seq.append(base)
                break
                
    return ''.join(seq)
```


```python
baseCumulativeFrequencies = getBaseFrequencies(libraryTable_complete)[1]
negList = []
numberToGenerate = 1000 #can generate many more; some will be filtered out by off-targets, and you can always select an arbitrary subset for inclusion into the library
for i in range(numberToGenerate):
    negList.append(generateRandomSequence(baseCumulativeFrequencies))
negTable = pd.DataFrame(negList, index=['non-targeting_' + str(i) for i in range(numberToGenerate)], columns = ['sequence'])

fqFile = 'temp_bowtie_input_negs.fq'
outputTempBowtieFastq(negTable, fqFile)
```


```python
#similar to targeting sgRNA off-target scoring, but looking for sgRNAs with 0 alignments
alignmentList = [(31,1,'~/indices/hg19.ensemblTSSflank500b','31_nearTSS_negs'),
                (21,1,'~/indices/hg19_maskChrMandPAR','21_genome_negs')]

alignmentColumns = []
for btThreshold, mflag, bowtieIndex, runname in alignmentList:

    alignedFile = 'bowtie_output/' + runname + '_aligned.txt'
    unalignedFile = 'bowtie_output/' + runname + '_unaligned.fq'
    maxFile = 'bowtie_output/' + runname + '_max.fq'
    
    bowtieString = 'bowtie -n 3 -l 15 -e '+str(btThreshold)+' -m ' + str(mflag) + ' --nomaqround -a --tryhard -p 16 --chunkmbs 256 ' + bowtieIndex + ' --suppress 5,6,7 --un ' + unalignedFile + ' --max ' + maxFile + ' '+ ' -q '+fqFile+' '+ alignedFile
    print bowtieString
    print subprocess.call(bowtieString, shell=True)

    #read unaligned file for negs, and then don't flip boolean of alignmentTable
    with open(unalignedFile) as infile:
        sgsAligning = set()
        for i, line in enumerate(infile):
            if i%4 == 0: #id line
                sgsAligning.add(line.strip()[1:])

    alignmentColumns.append(negTable.apply(lambda row: row.name in sgsAligning, axis=1))
    
alignmentTable = pd.concat(alignmentColumns,axis=1, keys=zip(*alignmentList)[3])
alignmentTable.head()
```

    bowtie -n 3 -l 15 -e 31 -m 1 --nomaqround -a --tryhard -p 16 --chunkmbs 256 ~/indices/hg19.ensemblTSSflank500b --suppress 5,6,7 --un bowtie_output/31_nearTSS_negs_unaligned.fq --max bowtie_output/31_nearTSS_negs_max.fq  -q temp_bowtie_input_negs.fq bowtie_output/31_nearTSS_negs_aligned.txt
    0
    bowtie -n 3 -l 15 -e 21 -m 1 --nomaqround -a --tryhard -p 16 --chunkmbs 256 ~/indices/hg19_maskChrMandPAR --suppress 5,6,7 --un bowtie_output/21_genome_negs_unaligned.fq --max bowtie_output/21_genome_negs_max.fq  -q temp_bowtie_input_negs.fq bowtie_output/21_genome_negs_aligned.txt
    0





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>31_nearTSS_negs</th>
      <th>21_genome_negs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>non-targeting_0</th>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>non-targeting_1</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>non-targeting_2</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>non-targeting_3</th>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>non-targeting_4</th>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
acceptedNegList = []
negCount = 0
for i, (name, row) in enumerate(pd.concat((negTable,alignmentTable),axis=1, keys=['seq','alignment']).iterrows()):
    oligo = upstreamConstant + row['seq','sequence'] + downstreamConstant
    if row['alignment'].all() and matchREsites(oligo, restrictionSites):
        acceptedNegList.append(('non-targeting_%05d' % negCount, 'negative_control', 'na', row['seq','sequence'], oligo, 0))
        negCount += 1
        
acceptedNegs = pd.DataFrame(acceptedNegList, columns = ['sgId', 'gene', 'transcript', 'protospacer sequence', 'oligo sequence', 'off-target stringency']).set_index('sgId')
```


```python
acceptedNegs.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gene</th>
      <th>transcript</th>
      <th>protospacer sequence</th>
      <th>oligo sequence</th>
      <th>off-target stringency</th>
    </tr>
    <tr>
      <th>sgId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>non-targeting_00000</th>
      <td>negative_control</td>
      <td>na</td>
      <td>GGCGCGGCGTGCAACCGGGA</td>
      <td>CCACCTTGTTGGGCGCGGCGTGCAACCGGGAGTTTAAGAGCTAAGCTG</td>
      <td>0</td>
    </tr>
    <tr>
      <th>non-targeting_00001</th>
      <td>negative_control</td>
      <td>na</td>
      <td>GCAGCCGGTAGGACACCGAC</td>
      <td>CCACCTTGTTGGCAGCCGGTAGGACACCGACGTTTAAGAGCTAAGCTG</td>
      <td>0</td>
    </tr>
    <tr>
      <th>non-targeting_00002</th>
      <td>negative_control</td>
      <td>na</td>
      <td>GGCGCGCCGTCTCCACTTTT</td>
      <td>CCACCTTGTTGGGCGCGCCGTCTCCACTTTTGTTTAAGAGCTAAGCTG</td>
      <td>0</td>
    </tr>
    <tr>
      <th>non-targeting_00003</th>
      <td>negative_control</td>
      <td>na</td>
      <td>GCGCGTTGCAGAGATAGACG</td>
      <td>CCACCTTGTTGGCGCGTTGCAGAGATAGACGGTTTAAGAGCTAAGCTG</td>
      <td>0</td>
    </tr>
    <tr>
      <th>non-targeting_00004</th>
      <td>negative_control</td>
      <td>na</td>
      <td>GATGTGGAGGCGTTGCCGCG</td>
      <td>CCACCTTGTTGGATGTGGAGGCGTTGCCGCGGTTTAAGAGCTAAGCTG</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
libraryTable_complete.to_csv(LIBRARY_WITHOUT_NEGATIVES_PATH, sep='\t')
acceptedNegs.to_csv(NEGATIVE_CONTROLS_PATH,sep='\t')
```

## Finalizing library design

* divide genes into sublibrary groups (if required)
* assign negative control sgRNAs to sublibrary groups; ~1-2% of the number of sgRNAs in the library is a good rule-of-thumb
* append PCR adapter sequences (~18bp) to each end of the oligo sequences to enable amplification of the oligo pool; each sublibary should have an orthogonal sequence so they can be cloned separately


```python

```
