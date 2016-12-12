# CRISPRiaDesign

This site hosts the sgRNA machine learning scripts used to generate the Weissman lab's next-generation CRISPRi and CRISPRa library designs [(Horlbeck et al., eLife 2016)](https://elifesciences.org/content/5/e19760). These are currently implemented as interactive scripts along with iPython notebooks with step-by-step instructions for creating new sgRNA libraries. Future plans include adding command line functions to make library design more user-friendly. Note that all sgRNA designs for CRISPRi/a human/mouse protein-coding gene libraries are included as supplementary tables in the eLife paper, so cloning of individual sgRNAs or construction of any custom sublibraries targeting protein-coding genes can simply refer to those tables. These scripts are primarily useful for the design of sgRNAs targeting novel or non-coding genes, or for organisms beyond human and mouse.

**To apply the exact quantitative models used to generate the CRISPRi-v2 or CRISPRa-v2 libraries**, follow the steps outlined in the Library_design_walkthrough (included as a Jupyter notebook or [web page](Library_design_walkthrough). 

To see full example code for de novo machine learning, prediction of sgRNA activity for desired loci, and construction of new genome-scale CRISPRi/a libraries, see the CRISPRiaDesign_example_notebook (included as Jupyter notebook or [web page](CRISPRiaDesign_example_notebook.md)).

###Dependencies
* Python v2.7
* Jupyter notebook
* Biopython
* Scipy/Numpy/Pandas
* Scikit-learn 
* bxpython (v0.5.0, https://github.com/bxlab/bx-python)
* Pysam
* [ScreenProcessing](https://github.com/mhorlbeck/ScreenProcessing)
External applications required:
* ViennaRNA
* Bowtie (not Bowtie2)
Large genomic data files required:
Links are to human genome files relied upon for the hCRISPRi-v2 and hCRISPRa-v2 machine learning--and required for the Library_design_walkthrough--but any organism/assembly may be used for design of new libraries or de novo machine learning.
* Genome sequence as FASTA ([hg19](http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/))
* FANTOM5 TSS annotation as BED ([TSS_human](http://fantom.gsc.riken.jp/5/datafiles/phase1.3/extra/TSS_classifier/))
* Chromatin data as BigWig ([MNase](https://www.encodeproject.org/files/ENCFF000VNN/), [DNase](https://www.encodeproject.org/files/ENCFF000SVY/), [FAIRE-seq](https://www.encodeproject.org/files/ENCFF000TLU/))
* HGNC table of gene aliases (not strictly required for the Library_design_walkthrough but useful in some steps)
* Ensembl annotation as GTF (not strictly required for the Library_design_walkthrough but used in other functions; release 74 used for the published library designs)