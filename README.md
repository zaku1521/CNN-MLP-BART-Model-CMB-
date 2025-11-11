# CNN-MLP-BART-Model-CMB-
IR2SMILES-BART: End-to-End SMILES Generation from Infrared (IR) Spectra
The CMB (CNN-MLP-BART) model is an innovative deep learning framework designed for the automated prediction of SMILES (Simplified Molecular Input Line Entry System) molecular structures directly from infrared (IR) spectra. This end-to-end model integrates a Convolutional Neural Network (CNN) for spectral feature extraction, a Multi-Layer Perceptron (MLP) for information vectorization, and a pretrained BART model for the generation of SMILES strings. Unlike traditional methods that require extensive prior chemical knowledge or spectral databases, CMB operates solely on raw IR data, eliminating the need for manual interpretation or subjective expertise in molecular analysis.
Top-1 SMILES Prediction Accuracy: 90% on QM9S dataset,Top-5 & Top-10 SMILES Prediction Accuracy: 93.5% and 95.5% respectively
Functional Group Recognition: 94% on QM9S, enabling precise identification of chemical groups from IR data
Cross-Dataset Validation: Achieving 83.2% Top-1 accuracy on the NIST dataset, demonstrating the model's robustness for real-world spectroscopic data.
Hardware and Environment Requirements 
CPU Node 
Processor (CPU): 4 cores (vCPUs), Intel(R) Xeon(R) Platinum 8200 series 
Processor (CPU): 8 cores (vCPUs) , Intel(R) Xeon(R) Platinum 8200 series 
Memory (RAM): 31 GB 
GPU: 1 × NVIDIA T4 (16 GB GDDR6 VRAM, Turing architecture, 2560 CUDA cores, supports 
Tensor Cores/RT Cores) 
Storage (Disk): 40 GB 
Use cases: Training the 3D-SAT-OLED model and performing large-scale inference
Dependencies 
Note: This project requires Python 3.9. Please ensure you are using a Python 3.9 environment before installing dependencies.





packages	version
transformers	4.41
tokenizers	0.19.1
sentencepiece	0.1.99
numpy	1.24
pandas	1.5
tqdm	4.66
lmdb	1.4.1
matplotlib	3.7
scikit-learn	1.0.2
Directory Structure
├── README.md
├── src/
├── train.lmdb
├── test.small.lmdb  
│   ├── bart.json                       
├── tokenizer-smiles-bart/ 
│   ├──vocab.json
│   ├──bart.json
│   ├──merges.txt
│   ├──special_tokens_map.json
├── cmb.py                  
├── functional group.py        
├── mcs.py                 
├──utils.py
generation                    
├── outputs/
│   ├── checkpoint-epoch-20-cnn.pt  
│   ├── final_model_cnn.pt                
├── results.csv                            
└── requirements.txt
Dataset
Properties: The QM9S dataset consists of molecular properties, including electronic and geometric features, with a focus on various quantum mechanical descriptors such as the energy, dipole moment, and total polarizability.
Format: The dataset is stored in .npz files containing:
1.Molecular coordinates
2.Atomic types
3.Target properties such as energies and dipole moments
Model
The model is CMB (CNN-MLP-BART), a deep learning architecture designed for predicting SMILES from infrared spectra. It integrates a Convolutional Neural Network (CNN) for feature extraction from IR data, followed by a Multi-Layer Perceptron (MLP) for vectorizing these features. The model uses the BART architecture for SMILES generation, leveraging a sequence-to-sequence framework to produce accurate molecular representations. This model is designed to handle spectral data and molecular structure predictions efficiently, enabling automated molecular identification directly from raw IR spectra.
Configuration


Quick Start
(1)Installation
pip install requirements.txt
(2) Train and Top-K accuracy
python cmb.py 
To run the training and evaluation code, the following steps are involved:Data ,as shown in the following figure('key' represents the data number, 'smi' represents the SMILES molecular formula, and 'ir' represents the infrared spectral data)

Training Data: The code requires the train.lmdb file, which contains the training dataset in LMDB format. This file must include molecular structures (SMILES) and corresponding infrared (IR) spectra.
Test Data: The test.small.lmdb file is used for evaluation after the training phase. Similar to the training data, this file should contain test samples with both IR spectra and ground-truth SMILES.
(3)Output
Model Checkpoints: After training for 20 epochs, the model will save the checkpoint at checkpoint-epoch-20-cnn.pt.
Final Model: The final trained model will be saved as final_model_cnn.pt in the outputs directory.
These outputs can be used for further inference or fine-tuning on additional datasets.
Accuracy Evaluation：During the evaluation phase, the code computes Top-K accuracy for the model's SMILES predictions. It checks if the correct SMILES appears within the top K predicted results. 
(4)Testdemo
After successfully running the demo. py code, 100 sets of IR spectral data from the test set will be output. as shown in the following figure.

Enter the IR spectral data number you want to test. 

Next, the standard SMILES molecular formula corresponding to the set of IR spectral data in the output dataset, the predicted top 10 SMILES molecular formulas, Top1 accuracy, Top5 accuracy, Top10 accuracy MCS、 Levinstein distance

(5)Maximum Common Substructure (MCS) Evaluation
This code evaluates the Maximum Common Substructure (MCS) accuracy by comparing the predicted SMILES strings with each other and measuring how much structural similarity exists between them. It calculates the MCS between each pair of SMILES, which is the largest shared substructure found in both molecules. The accuracy is determined by the ratio of the size of the MCS to the size of the individual molecules. The highest MCS accuracy for each SMILES is stored in a new column and saved to a CSV file.
To run the code, simply execute the script, which will:
1.Load the predicted SMILES from smiles_output.csv.
2.Compute the MCS for each pair of SMILES in the dataset.
3.Calculate the average MCS accuracy for each molecule.
(6)functional group recognition Evaluation
This code evaluates the functional group recognition accuracy by comparing the predicted SMILES strings (from IR spectra) with the ground-truth SMILES in the dataset. It utilizes SMARTS patterns to detect specific chemical substructures (functional groups) within the molecules. For each molecule, the code checks if the predefined functional groups—such as Alcohol, Carboxylic Acid, Ketone, and others—are present using RDKit’s substructure matching capabilities. A binary score (0 or 1) is assigned for each functional group based on whether it is detected in the molecule.
To run the code, simply execute the script, and it will automatically:
1.Load the predicted and ground-truth SMILES from resultsALL.csv and smiles_outputALL.csv.
2.Match the functional groups using the provided SMARTS patterns.=
3.Save the results (a DataFrame of 0s and 1s for each functional group) in the functional_groups_match.csv file for further analysis.
(7)Test Demo
After running the python demo.py command, the program will display 3 IR spectra with corresponding indices. You can input the index of a specific IR spectrum to trigger the model to perform inference using the /outputs/checkpoint-epoch-20-cnn.pt checkpoint. The model will then output the predicted Top-1 to Top-10 SMILES molecular structures.
Additionally, the program will compare these predictions with the true SMILES from the test.small.lmdb file for the selected IR spectrum index. Based on this comparison, the model will calculate and output the Top-1, Top-5, and Top-10 accuracy.
This allows you to evaluate the model's performance on a given test sample in terms of how well it ranks the correct SMILES among the predicted top-k results.
Expected Results 
Metric	Value
Top-1 Accuracy	90%
Top-5 Accuracy	93.5%
Top-10 Accuracy	95.5%
Functional Groups (Top-1 SMILES)	94%
MCS Accuracy (Top-1 SMILES)	83.4%
