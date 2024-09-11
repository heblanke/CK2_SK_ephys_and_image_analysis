# Welcome!
![action_potential](https://github.com/heblanke/VTA_DA_neurons_casein_kinase_2_and_SK_channels-/assets/97766161/02929cb6-848a-4b0f-9b6a-c3b4257765a9)

## Disclaimer ##
This repository is not under constant development. Errors may arise as a result of dependency conflicts. Use requires basic Python knowledge and familiarity with Jupyter notebooks in order to run the examples. 


# Python code to analyze single cell, patch clamp electrophysiology data collected in Axograph
code for the 3xTg small conductance calcium activated potassium (SK) channel - casein kinase 2 (CK2) study

###
The example directory houses example code workflows and files. They are effectively plug and play. In order to run the analysis: Download the entire directory (https://docs.github.com/en/repositories/working-with-files/using-files/downloading-source-code-archives#), confirm your installment of Jupyter notebooks (https://jupyter.org/install), and run the code within the notebook. It is entirely self contained, and will output data to your current working directory. Given this project is not under constant development, it is possible that you will experience conflicts between packages. 

###
These examples are not intended to represent every use case, but as a representation of how Axograph files were analyzed during the writing of the manuscript. 

###
Unfortunately, GitHub has a file size cap at 25MB. Thus, the files in the "spontaneous_firing_analysis" directory will need to be unzipped prior to analysis. 


###
All functions to analyze data are housed in a Jupyter notebook. Code can be run by installing Jupyter and opening the .ipynb file. Code is appropiately commented to indicate when functions work on directories or on single objects. 

###
All functions for the paper are housed in the .py file in the main branch. They are verbosely commented, and the notebook examples of code call from the .py file, which requires it be installed in your current working directory or that you point to it using "sys.path.append('where_the_.py_file_is_housed')" 


###
If you are unfamiliar with how code download from GitHub work, from the main brain, in the upper right hand corner click the green "Code" dropdown then "Download Zip" or follow the link included above. 






![wt_inject_gif](https://github.com/heblanke/VTA_DA_neurons_casein_kinase_2_and_SK_channels-/assets/97766161/95505bfd-2399-446a-a7af-4071528ea589)





![image](https://github.com/user-attachments/assets/2c27bd4b-53db-488f-974e-9ce4ab181ffa)


###
Image analysis

###
The entire directory for image analysis can be downloaded and run locally. It relies on Cellpose v3, which has some backend requirements like PyTorch. It is necessary to pip install cellpose. Follow their documentation (https://github.com/mouseland/cellpose). The analysis represented here begins after the extraction of either the sum intensity projection or the brightest slice from the channel to be analyzed, and thus the example images for segmentation are .tif files rather than .czi files. To arrive at these from your file of interest, pull them in with your respective image handler, for .czi it is CziReader, which converts them to Numpy arrays, then either slice for brightest or convert to higher bit rate, and sum into a single plane. 

###
Segentation occurs on TH positive cells, calculation occurs on the channel that is specified in the code. Check documentation within the notebooks and the image_analysis.py file to see exactly how its set up. The Examples are extracted single 1µm section tiffs. If paths are pointing to the correct place, it should function as plug and play. This has not been adequately tested on a Windows machine. 
