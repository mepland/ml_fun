# ml\_fun
Matthew Epland, PhD  
[phy.duke.edu/~mbe9](http://www.phy.duke.edu/~mbe9)  

Holds small machine learning projects from my desktop, using pytorch and CUDA.  

## Cloning the Repository
ssh  
```bash
git clone --recurse-submodules git@github.com:mepland/ml_fun.git
```

https  
```bash
git clone --recurse-submodules https://github.com/mepland/ml_fun.git
```

## Installing Dependencies
It is recommended to work in a [python virtual environment](https://realpython.com/python-virtual-environments-a-primer/) to avoid clashes with other installed software. If you do not wish to use a virtual environment you can usually just run the first few cells of the notebook in question - useful when running on cloud-based virtual machines.
```bash
python -m venv ~/.venvs/ml_fun
source ~/.venvs/ml_fun/bin/activate
pip install -r requirements.txt
```

## Running the Notebook

```bash
jupyter lab notebook_name.ipynb
```
