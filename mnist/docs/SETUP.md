# MNIST project setup

This guide walks through setting up the virtual environment and dependencies for the MNIST PyTorch project.

## Prerequisites

- **Python 3.8+** (3.10 or 3.11 recommended)
- `pip` (usually included with Python)

Check your version:

```bash
python3 --version
pip --version
```

## Step 1: Go to the project folder

```bash
cd path/to/kecInf/mnist
```

Use the actual path to your `mnist` folder (e.g. `cd /Users/userName/Desktop/kecInf/mnist`).

## Step 2: Use or create the virtual environment

### Option A — venv already exists

If `mnist/venv` is already there (e.g. after someone created it for you), activate it:

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your prompt.

### Option B — create a new venv

If there is no `venv` folder or you want a fresh one:

```bash
python3 -m venv venv
source venv/bin/activate   # or the Windows equivalent above
```

## Step 3: Install dependencies

With the virtual environment **activated**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

- `torch` — PyTorch
- `torchvision` — datasets (including MNIST) and image transforms

## Step 4: Verify

Download MNIST and print basic info:

```bash
python load_mnist.py
```

You should see something like:

```
Setting up MNIST...
Train samples: 60000
Test samples:  10000
Batch shape: images torch.Size([64, 1, 28, 28]), labels torch.Size([64])
MNIST is ready.
```

The first run will download the dataset into `mnist/data/`.

## Leaving the virtual environment

When you’re done working in this project:

```bash
deactivate
```

Your prompt returns to normal. Reactivate with `source venv/bin/activate` (or the Windows equivalent) next time you work on this project.

## Troubleshooting

| Issue                            | What to try                                                                                                 |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `python3: command not found`   | Use `python` instead of `python3`, or install Python from [python.org](https://www.python.org/downloads/). |
| `pip install` fails or is slow | Use a mirror or install behind a proxy: e.g.`pip install -r requirements.txt -i https://pypi.org/simple`  |
| Permission errors                | Don’t use `sudo` with the venv. Run all commands with the venv activated and without `sudo`.           |
| Script runs but imports fail     | Ensure the venv is activated and you ran `pip install -r requirements.txt` in that same environment.      |
