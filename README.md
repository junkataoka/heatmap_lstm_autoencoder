# Required packages
1. PyTorch (GPU Enabled)
2. PyTorch Lightning 
3. numpy
4. pandas
5. matplotlib
6. seaborn


# How to Run
1. Install required packages
2. run preprosses.py to generate dataset/input.pt and dataset/target.pt files
3. run main.py with or without arguments (Detailed arguments are given in main.py file)
4. run main.py with --test argument enabled to generate prediction as dataset/#_pred.pt
5. run postprosses.py to generate csv files