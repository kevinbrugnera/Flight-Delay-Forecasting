# Environment Setup

Create and configure the Python environment required to run the project.

## First Time Set-Up

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Upgrade `pip` and install all required packages listed in `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Standard Workflow

### 1. Use the environment in Jupyter Notebook

Activate the environment:

```bash
source .venv/bin/activate
```

Install the Jupyter kernel:

```bash
pip install ipykernel
```

Register the virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name project-env --display-name "Python (project-env)"
```

Then select the kernel: **Kernel → Change Kernel → Python (project-env)**

The notebook will now use the packages installed in the project environment.

### 2. Run Python scripts using the environment

Activate the environment:

```bash
source .venv/bin/activate
```

Then run:

```bash
python script_name.py
```

The script will now use the packages installed in the project environment.
