# License-Plate-Detector
License Plate Detector implementation using OpenCV and EasyOCR.   
Was built as part of APU IPCV module assignment   

## Installation
- Install Requirements   
  All requirements are listed inside requirements.txt. To install use the command:
  `pip install -r requirements`
- Jupyter Notebook   
  To open the `.ipynb` file. It is advised to open it using jupyter notebook which can be installed with the command:
  `pip install jupyter`

## Usage
- Jupyter Notebook   
  The Jupyter notebook or `.ipynb` file is used to give a step-by-step flow of detection process.  Each process will be done in separate box and each process output will be shown
- LPD script    
  LPD script or `LPD.py` can be used as a script to detect license plate. It works based on arguments listed below:
  ```bash
  $ python3 LPD.py -h
  usage: LPD.py [-h] [-i INPUT] [-s] [-d] [-v]

  Simple License Plate Detection using OpenCV and EasyOCR

  optional arguments:
    -h, --help            show this help message and exit
    -i INPUT, --input INPUT
                          path to an image
    -s, --show            show final visualizations
    -d, --debug           show all image processing steps
    -v, --validate        validate result with test images

  ```
  This help page can be toggled by the command `python LPD.py -h`.
  
## Validation
The system was tested against 25 images which contains a license plate with the result below:
```bash
==========Validation Result===========

Correctly Detected: 20
Total Image Tested: 25
Accuracy: 0.8

======================================
```
This result can be produced by running the script using `-v` or `--validate` option.
