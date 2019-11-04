# CSE6242 Project ReadMe

## Forest Classifier information
There are two files called "randomForestClassifer", one with extension .py and another with extension .ipynb. The former is a normal python file to be run locally on your laptop/computer. The latter is a jupyter notebook to be ran on Google Colaboratory.

### Local Random Forest Classifier information

#### How to run randomForestClassifier.py
1. Make sure you have the latest randomForestClassifier.py
2. cd to the directory with randomForestClassifier.py in it
3. make sure the util.py and csv file you want to process are in the same directory
4. Make sure file_name string in the .py file is set to the name of the csv file you want to process
5. run the following command on the command line: python3 randomForestClassifier.py

#### Further info on randomForestClassifier.py
The plots, including ROC, confusion matrices, and f1 score graphs, will be generated as png files in the directory you are running the python file from.
To change the hyperparameters used, go into randomForestClassifier.py and update the lists under the comment that says "global variables"

### Jupyter Notebook Random Forest Classifier information

#### How to run randomForestClassifier.ipynb
1. Make sure you have a google account
2. log into your google account
3. open Google Drive
4. Set up a jupyter Notebook (See Google_GPU.pdf under the "Other" folder in our shared Google Drive for this project)
5. Put the randomForestClassifier.ipynb file somewhere in your Google Colab folder wherever you want to run it
6. Open the ipynb file in Google Colab
7. Click on the ">" tab under the "+Code" button at the top left
8. Click on "Upload"
9. Upload the data file to be processed and util.py
10. Make sure file_name string is set to the name of the file you want to process in the ipynb file.
11. Run the script
12. It will install TensorFlow GPU 2.0.0. It will say in red that to use this new version you must restart the runtime with a button that says restart on it. Press the button and wait for the runtime to be reinitialized
13. Run the script again.
14. It will ask you to go to a url to put in the authorization code. Go to that url, choose your gmail account, press "Allow", and copy the url shown on screen.
15. paste the url in the place it asks to and press Enter key. The script should run now

Note: When the instructions say "run the script", please run the first section that has only two lines of code first, then the next section that has the main code.

#### Further info on randomForestClassifier.ipynb
The png file plots will be generated in "drive/My Drive/Colab Notebooks/CSE6242Project/". You may need to create the CSE6242Project folder before running this.
To change the hyperparameters used, go into randomForestClassifier.ipynb and update the lists under the comment that says "global variables"
