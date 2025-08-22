# CASA0010 Dissertation Code – He Muheng

This repository contains all code, environment files, and scripts used for my MSc Dissertation project:
**“Relationship between Transport Accessibility and Property Prices in London: A Machine Learning Approach with Socioeconomic Considerations.”**

------------------------------------------------------------------------

## ⚠️ Important Notice

During development, **absolute paths** were used, and I apologize for the inconvenience this may cause.

Please execute it first after cloning：

bash：

git lfs install

git lfs pull

To ensure the project runs correctly, it is recommended to manually create the following directory on your local machine:

/Users/muhenghe/Documents/BYLW

Then copy the following two folders from this repository into that path:

-   `start` (obtained by unzipping `start.zip`)
-   `项目初始`

After that, navigate to:

/Users/muhenghe/Projects/bylw-pipeline/项目初始/pythonProject2

and run the Python scripts sequentially in the following order:

python "1.hex.py"

python "2.hex.py"

python "3. accessibility score.py"

python "4.sev and dwelling density.py"

python "5. price.py"

python "6.price2.py"

python "7.ml1 .py"

python "8.ml1.py"

python "9.ml1.py"

python "10.ml1.py"

python "11.ml1.py"

python "12.ml1 .py"

## Quick Start

1.  Clone the repository

2.  Setup symlinks to mirror absolute paths

bash:

./setup_symlinks.sh

3.  Create Conda environment

bash:

conda env create -f environment.yml conda activate bylw

4.  Fetch datasets The dataset is provided as a compressed file start.zip (\~328MB) stored via Git LFS. Please make sure you have Git LFS installed:

bash:

git lfs install git lfs pull

Unzip start.zip in the project root to restore the start/ folder.

5.  Run the full pipeline

bash:

./run_all.sh
