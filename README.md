MSThunder provide a deep learning-based nontargeted analytical framework for the accurate and rapid identification of unknown organic pollutants in water. The training data, code, model parameters and candidates used for training and prediction are available through our experiments conducted on an Ubuntu 20.04 environment. The interface mode of MSThunder is available by decompressing in the file in RAR format (msthunder.rar and msthunderfile.rar) (doi.org/10.5281/zenodo.12602805). The files of “msthunderfile.rar” need to be unzipped under the folder of the “msthunder”. A case file named “Pesticides” can be run in the Windows environment equipped with 16 GB of RAM and a 2 GB NVIDIA GPU. Due to environment configuration issues, the current version does not yet support offline processing of raw data. If you want to analyze other data on UPLC-HRMS, you can send the raw file to our e-mail (qzliu@rcees.ac.cn) or update it to the Zenodo, GNPS or other online databases. We are currently developing the relevant functionality and will make it available online as soon as possible. The current version is compatible with ThermoFisher, Agilent, and other vendors whose raw data can be converted via MSConvert. Then, you can subsequently analyze the data using MSThunder after our process the raw data in a Linux system and return the converted file. The detailed usage method of MSThunder is as follows. We also provide a tutorial video named “Guideline.mp4” for the use of MSThunder.

First, the batch-processed files need to be placed in the MSThunder directory. 

Input button: Enter the file name into the blank space before "Input", click the "Input" button, and select the ion mode ("Positive" or "Negative") to obtain the batch-processed file information. 

Precursor/RT button: Enter the specified "precursor/retention time" to search for the TIC of the precursor ion and the MS2 information at that retention time; enter the specified "precursor" to search for the TIC of the precursor ion and the MS2 information at the earliest retention time. 

SMILES button: Enter the specified SMILES to view the compound structure diagram in Structure. 

Scale button: It can obtain all MS1 spectra at the current retention time and display them in MS1.

MS2 List: Click on any row of data to get the corresponding TIC, MS2, and Structure information. Double-click on any row of data to view the precursor, retention time, ISMILES, and candidate precursor formula in the Candidate section. 

Candidate: Double-click the retention time to obtain the MS1 spectrum of the corresponding precursor at the current retention time. Double-click the precursor formula to execute the structure prediction function for that molecular formula; the candidate score information will be displayed in Ranking. If a reference spectrum is found among the top 10 candidates, the spectrum matching result will be shown in MS2-candidate. Tips: If the candidate precursor molecular formula needs to be replaced, you can enter the replacement formula in the blank space of Formula and then double-click to perform the prediction. 

Ranking: Double-click any piece of data to display the structure information diagram in the Structure section. 

Note: If the image display encounters issues, it may be fixed to decrease the computer's display ratio.

The model and interface of MSThunder were written by Python 3.10.5 with associated packages, consisting of PySide6 6.6.0, torch 1.13.0+cu116, numpy 1.23.4, matplotlib 3.6.2, rdkit 2023.9.2, adjustText 0.8, tqdm 4.64.1, pubchempy 1.0.4 and pandas 1.5.0.
