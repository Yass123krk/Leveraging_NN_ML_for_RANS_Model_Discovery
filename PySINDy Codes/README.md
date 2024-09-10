# OutBound's Project Setup Instructions
Welcome to the OutBound's Project ! Here are the instructions to set up your environment to run the code. You have two methods to choose from:

## Method 1: Using the Pre-existing Virtual Environment
### Step 1: Activate the Virtual Environment
Navigate to the project directory and activate the virtual environment:

**Windows:**

.venv\Scripts\activate

**macOS and Linux:**

source .venv/bin/activate

### Step 2: Verify the Environment
Ensure the virtual environment is activated. You should see the virtual environment name at the beginning of your terminal prompt.

## Method 2: Setting Up Manually Using requirements.txt
### Step 1: Create a New Virtual Environment
Create a virtual environment to house the project's dependencies:

**Windows:**
python -m venv .venv

**macOS and Linux:**
python3 -m venv .venv

### Step 2: Activate the Virtual Environment
Use the commands provided in Method 1 to activate the new virtual environment.

### Step 3: Install Dependencies
With the environment activated, install the required dependencies:
pip install -r requirements.txt

### Step 4: Verify Installation
Check that all packages have been installed:
pip list

## Running the Code
With the environment set up, you can run the scripts or notebooks. If using Jupyter Notebook:

### Ensure Jupyter Notebook is installed in your virtual environment:
pip install notebook

### Start Jupyter Notebook:
jupyter notebook

### Navigate to the notebook file through the Jupyter interface and run the cells sequentially.

## Data Handling and Analysis
After setting up your environment and dependencies, you can proceed with loading and manipulating the data:


## Specific Instructions for Notebooks

### FOR PySINDy_k_Approach.ipynb

1-Launch the PySINDy_k_Approach.ipynb Jupyter Notebook.
2-Start by running the pip install cells if those libraries are not installed in your environment.
3-Run the cell to import the libraries.
4-Load the unnormalized data. This dataset has been processed with smoothing and data augmentation.
5-Run the training cell (can take up to 1 minute).
6-Choose to run the testing and validating cells of interest, including various predictions and covariances computations.

### FOR PySINDy_Final_Approach.ipynb

1-Launch the PySINDy_Final_Approach.ipynb Jupyter Notebook.
2-Start by running the pip install cells if those libraries are not 3-installed in your environment.
Run the cell to import the libraries.
4-Choose to load either normalized or unnormalized data. Adjust plotting instructions based on your choice.
5-Set up and run training cells for different dynamics equations.
6-Adjust parameters for desired equations before plotting.
7-Run the cell for plotting normalized and averaged results directly.