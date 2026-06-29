# Chord-DS-HPS-NN
Project: Evaluation of Downsampling and Harmonic Product Spectrum Parameters for Neural Network-Based Chord Recognition

Software: Python

Dataset
=======
Dataset Name: Chord-3K5

The Chord-3K5 dataset consists of 3,500 audio samples representing seven major guitar chord classes (C, D, E, F, G, A, and B). Each sample is a 2-second WAV audio signal with a sampling frequency of 40 kHz.
Dataset Split: Training: 2,450 samples (350 samples per class). Validation: 525 samples (75 samples per class). Testing: 525 samples (75 samples per class)

The Chord-3K5 dataset is an augmented version of the original dataset. Data augmentation was performed using pitch shifting and time stretching implemented with the librosa Python package.

Original Dataset
The original dataset consists of 140 audio samples from the same seven chord classes.
Dataset Split: Training: 98 samples (14 samples per class); Validation: 21 samples (3 samples per class); Testing: 21 samples (3 samples per class)

The original recordings have a duration of 3 seconds and were sampled at 40 kHz. All recordings were performed using a Yamaha CPX-500-II guitar with a single playing variation for each chord. The recordings were conducted in a quiet environment to minimize background noise, and all audio files are single-channel (mono) recordings.

Augmented WAV File Naming Convention
====================================
Augmented audio files follow the naming format:

V_gWXX_psYY_tsZ.ZZ.wav

Meaning: V:Target chord label; W: Source chord label; XX: (Source sample number); YY: Pitch shift (in semitones); Z.ZZ: Time-stretch factor.

Example
a_gb01_ps-2_ts1.00.wav

Meaning: a: Target chord label: A; b: Source chord label: B; 01: Source sample number 01; -2: Pitch shift of −2; semitones; 1.00: Time-stretch factor of 1.00

