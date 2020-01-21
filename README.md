# ml\_fun
Matthew Epland, PhD  
[phy.duke.edu/~mbe9](http://www.phy.duke.edu/~mbe9)  

Holds machine learning projects built on my desktop, using pytorch and CUDA.  

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

Note, to fully use ipywidgets (necessary for tqdm.notebook) you must also have nodejs installed and run some jupyter commands. See [here](https://ipywidgets.readthedocs.io/en/stable/user_install.html) for full details, basic commands below:
```bash
# conda install -c conda-forge nodejs # install nodejs on your system. If you are using conda, can get from conda-forge
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## Running Notebooks

```bash
jupyter lab notebook_name.ipynb
```
