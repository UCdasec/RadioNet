# RadioNet

**ADA** contains code for Adversary Domain Adptation. `af_classifier.py` is the main function for ADA.

**triplet_network** contains code for Triplet Network. `triplet_training.py` is the main function for triplet network.

`model_training.py` for the same-day or cross-day scenario evaluation.

`finetune.py` for fine-tuning pre-trained models. 

## Defination of parameters

We use the same definations of parameters for all those functions. Currently, you can modify those parameters at the entrance of each function (such as "testOpts()" or "main()"):

`dataPath`: Path for Training dataset (Day1 dataset).

`testPath`: Path for Test dataset (Day 2 dataset for the cross-day scenario).

`location`: Specify the location of collected dataset, such as "before_fft", "after_fft" and "symbols".

`slice_len`: length for each slice, such as 288.

`dataType`: Data representation. "IQ" for I/Q data rapresentation; "spectrogram" for spectrogram data representation.

`num_slice`: Number of slices to extract from each device.

`start_idx`: Start index to extract slices.

`stride`: Stride when cutting slices, such as 144 or 288.

`modelType`: Model architecture, such as "homegrown" and "DF".

## Usage 
After you modify the parameters, run `python *.py` to train and test your models. 

