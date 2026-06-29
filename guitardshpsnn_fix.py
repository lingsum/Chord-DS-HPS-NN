# Evaluation of Downsampling and Harmonic Product Spectrum Parameters
# for Neural Network-Based Chord Recognition

import numpy as np
import os
from scipy.fftpack import fft
from scipy.io import wavfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import glob
from sklearn.metrics import precision_score, recall_score, f1_score

# =============================================================
# Class and Function Definitions
# =============================================================
# Define NN Model
class NNet(nn.Module):
  
    # One hidden layer
    def __init__(self, InputSize, HiddenSize, OutputSize):
        super().__init__()

        self.fc1 = nn.Linear(InputSize, HiddenSize)
        self.fc2 = nn.Linear(HiddenSize, OutputSize)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)

        return x

# -------------------------------------------------------------
def prepro(wav0, DSFactor, HPSLevel):
    # Data preprocessing
    
    # Read wav
    fs, x0 = wavfile.read(wav0)
 
    # Truncating 2^m data
    x1 = x0[0:NumData]
    
    # Downsampling
    if DSFactor == 1:       
        x2 = x1
    elif DSFactor == 2:
        x2 = x1[0::2]       
    elif DSFactor == 4:
        x2 = x1[0::4]       
    elif DSFactor == 8:
        x2 = x1[0::8]       
    elif DSFactor == 16:
        x2 = x1[0::16]
    elif DSFactor == 32:
        x2 = x1[0::32]      
    elif DSFactor == 64:
        x2 = x1[0::64]      
    elif DSFactor == 128:
        x2 = x1[0::128]     
    else:
        x2 = x1[0::256]     
    
    # Normalization
    max1 = max(abs(x2))
    y0 = x2/max1
    
    # Windowing
    window = np.hamming(len(y0))
    y1 = y0*window

    # FFT
    # Take left half portion of magnitude values
    Ly1 = len(y1)
    y2 = abs(fft(y1))
    y2[0] = 0
    y2 = y2[0:int(Ly1/2)]

    # HPS
    for k in range(HPSLevel):
        ya = y2[::2]
        yb = y2[0:int(len(ya))]
        y2 = ya * yb
         
    # Replace inf (overflow data) with very large number (1x10^5)
    y2[y2 == np.inf] = 1e5
    
    # Logarithmic trasformation
    # Add '1' to avoid infinity
    y3 = np.log10(y2+1)
    
    return y3

# -------------------------------------------------------------
# Prepare data
def prepdata(directory, DSFactor, HPSLevel):
    # Find all WAV files in the specified directory
    wav_files = sorted(glob.glob(os.path.join(directory, "*.wav")))
    if not wav_files:
        raise ValueError(f"No WAV files found in {directory}")

    # Determine frame size from first file
    first_pre = prepro(os.path.abspath(wav_files[0]), DSFactor, HPSLevel)
    frame = len(first_pre)

    # Allocate output array
    data2 = np.zeros((len(wav_files), frame))

    # Process all files
    for k, wav_file in enumerate(wav_files):
        data2[k, :] = prepro(os.path.abspath(wav_file), DSFactor, HPSLevel)

    return data2

# -------------------------------------------------------------
# Datatrain datavalid and datatest
def dataAll(DSFactor, HPSLevel):
    # Datatrain: 350 samples/class (2450 samples total)
    rc = prepdata('training/c', DSFactor, HPSLevel)
    rd = prepdata('training/d', DSFactor, HPSLevel)
    re = prepdata('training/e', DSFactor, HPSLevel)
    rf = prepdata('training/f', DSFactor, HPSLevel)
    rg = prepdata('training/g', DSFactor, HPSLevel)
    ra = prepdata('training/a', DSFactor, HPSLevel)
    rb = prepdata('training/b', DSFactor, HPSLevel)
    dtrain = np.vstack((rc, rd, re, rf, rg, ra, rb)) # vertical stack
 
    # Datavalid: 75 samples/class (525 samples total)
    vc = prepdata('validation/c', DSFactor, HPSLevel)
    vd = prepdata('validation/d', DSFactor, HPSLevel)
    ve = prepdata('validation/e', DSFactor, HPSLevel)
    vf = prepdata('validation/f', DSFactor, HPSLevel)
    vg = prepdata('validation/g', DSFactor, HPSLevel)
    va = prepdata('validation/a', DSFactor, HPSLevel)
    vb = prepdata('validation/b', DSFactor, HPSLevel)
    dvalid = np.vstack((vc, vd, ve, vf, vg, va, vb))  # vertical stack
    
    # Datatest: 75 samples/class (525 samples total)
    sc = prepdata('testing/c', DSFactor, HPSLevel)
    sd = prepdata('testing/d', DSFactor, HPSLevel)
    se = prepdata('testing/e', DSFactor, HPSLevel)
    sf = prepdata('testing/f', DSFactor, HPSLevel)
    sg = prepdata('testing/g', DSFactor, HPSLevel)
    sa = prepdata('testing/a', DSFactor, HPSLevel)
    sb = prepdata('testing/b', DSFactor, HPSLevel)
    dtest = np.vstack((sc, sd, se, sf, sg, sa, sb))  # vertical stack
    
    return (dtrain, dvalid, dtest)

# -------------------------------------------------------------
def trainvaltest(DSFactor, HPSLevel, NumEpochs):

    # -------------------------------
    # Load data
    # -------------------------------
    XTrain, XVal, XTest = dataAll(DSFactor, HPSLevel)

    # Labels
    # Number of class: 7
    # Training: 350 samples/class 
    # Validation: 75 samples/class 
    # Testing: 75 samples/class 

    YTrain = np.repeat(np.arange(7), 350)
    YVal   = np.repeat(np.arange(7), 75)
    YTest  = np.repeat(np.arange(7), 75)

    # -------------------------------
    # Convert to tensors
    # -------------------------------
    XTrain = torch.FloatTensor(np.array(XTrain))
    XVal   = torch.FloatTensor(np.array(XVal))
    XTest  = torch.FloatTensor(np.array(XTest))

    YTrain = torch.LongTensor(YTrain)
    YVal   = torch.LongTensor(YVal)
    YTest  = torch.LongTensor(YTest)

    # -------------------------------
    # Determine NN dimensions automatically
    # -------------------------------
    InputSize  = XTrain.shape[1]     # Number of features
    OutputSize = 7                   # Guitar chord c,d,e,f,g,a,b
    HiddenSize = (InputSize + OutputSize) // 2

    # -------------------------------
    # Datasets
    # -------------------------------
    TrainDataset = TensorDataset(XTrain, YTrain)
    ValDataset   = TensorDataset(XVal, YVal)
    TestDataset  = TensorDataset(XTest, YTest)

    # -------------------------------
    # DataLoaders
    # -------------------------------
    TrainLoader = DataLoader(
        TrainDataset,
        batch_size=BatchSize,
        shuffle=True
    )

    ValLoader = DataLoader(
        ValDataset,
        batch_size=BatchSize,
        shuffle=False
    )

    TestLoader = DataLoader(
        TestDataset,
        batch_size=BatchSize,
        shuffle=False
    )

    # -------------------------------
    # Model
    # -------------------------------
    torch.manual_seed(seed)
    model = NNet(InputSize, HiddenSize, OutputSize)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LearningRate
    )

    # -------------------------------
    # Training
    # -------------------------------
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(NumEpochs):
        model.train()
        for inputs, labels in TrainLoader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():

            for inputs, labels in ValLoader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = 100.0 * correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    # -------------------------------
    # Load best model
    # -------------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # -------------------------------
    # Final Test
    # -------------------------------
    model.eval()
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():

        for inputs, labels in TestLoader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_acc = 100.0 * correct / total

    # Multi-class metrics
    precision = 100 * precision_score(
        all_labels,
        all_preds,
        average='macro',     # balance samples/class
        zero_division=0
        )

    recall = 100 * recall_score(
        all_labels,
        all_preds,
        average='macro'     # balance samples/class
        )

    f1 = 100 * f1_score(
        all_labels,
        all_preds,
        average='macro'     # balance samples/class
        )

    return (InputSize, best_val_acc, test_acc, precision, recall, f1)

# =============================================================
# MAIN PROGRAM
# =============================================================
# Suppress overflow warning for multiply
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Fixed parameters
NumData = 65536         # Number of raw data to be processed (2^n)
BatchSize = 32          # Batch size for dataset
seed = 42               # Fixed random seed
LearningRate = 0.001    # Default value for Adam optimizer

# Evaluate DSFactor and HPS Level
NumEpochsList = [50,100,150]
DSFactorList = [1,2,4,8,16,32,64,128,256]
for NumEpochs in NumEpochsList:
    print(f'NumEpochs = {NumEpochs}')
    
    for DSFactor in DSFactorList:
        if DSFactor == 1:
            X = 4; Y = 12
        elif DSFactor == 2:
            X = 3; Y = 11
        elif DSFactor == 4:
            X = 2; Y = 10
        elif DSFactor == 8:
            X = 1; Y = 9
        elif DSFactor == 16:
            X = 1; Y = 8
        elif DSFactor == 32:
            X = 1; Y = 7  
        elif DSFactor == 64:
            X = 1; Y = 6
        elif DSFactor == 128:
            X = 1; Y = 5
        else:               # DSFactor = 256
            X = 1; Y = 4
  
        for HPSLevel in range(X,Y):
            # Train and test neural network                             
            (FDimension, valAcc, testAcc, precision, recall, f1
             ) = trainvaltest(DSFactor,HPSLevel,NumEpochs)
            
            # Display of the results
            print(f'DSFac = {DSFactor}; HPSLev = {HPSLevel};',
                  f'FDim = {FDimension}; BVAcc = {valAcc:.2f} %;',          
                  f'TAcc = {testAcc:.2f}%; Prec = {precision:.2f}%;',
                  f'Rec = {recall:.2f}%; F1 = {f1:.2f}%')
        print('')
    



