# Project Setup

Set up the virtual environment for Orenda.

## Prerequisites

- Python 3.12.x or 3.11.9 on your system DO NOT USE other python versions
- Git (for cloning the repository)
- Make sure you're in the model_demos directory
```cmd
   cd model_demos
```
## Check Python Version
### Windows
   ```cmd
   py --version
   ```
### OS/Linux
   ```cmd
   python --version
   ```

## Virtual Environment Setup

### Windows

1. **Create the virtual environment:**
   ```cmd
   py -m venv orenda
   ```

2. **Activate the virtual environment:**
   ```cmd
   orenda\Scripts\activate
   ```

3. **Verify activation:**
   You should see `(orenda)` at the beginning of your command prompt.

### Mac/Linux

1. **Create the virtual environment:**
   ```bash
   python3.12 -m venv orenda
   ```

2. **Activate the virtual environment:**
   ```bash
   source orenda/bin/activate
   ```

3. **Verify activation:**
   You should see `(orenda)` at the beginning of your terminal prompt.

## Installing Dependencies
Once your virtual environment is activated, install the required packages:

Torch Libs for use with Nvidia 5000 series needs to be from the nightly build at the moment (Sept 2025). For those machines/GPU cards use

```bash
    pip install -r pytorch-cu128-requirements.txt
```
Else non-5000 series Nvida card OR mac/linux use:

```bash
    pip install -r pytorch-requirements.txt
```

then get the rest of the requirements.

```bash
    pip install -r requirements.txt
```
## Windows only Last Step libvips 
### Moondream2 Issue without libvips
We need to add libvips library to the system path on Windows to use moondream2.  The bin zip is already in this repo in the model_demos/dependencies/libvips/vips-dev-w64-web-8.17.2.zip you can unzip in place and add to system PATH, or run the helper function to automatically do that
```bash
python libvips_win_helper.py
```
## Pre-Load Models
All dependencies should be in. Restart terminal just to make sure (more of a windows issue), then download all the models
```bash
python model_loader.py
```
Done!

## Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

### Python Command Not Found (Windows)
If `python` is not recognized, try using `py` instead (typical for Windows):
```cmd
py -m venv orenda
```

### Permission Issues (Mac/Linux)
If you encounter permission issues, you might need to use `python3` instead of `python`:
```bash
python3 -m venv orenda
```

### Virtual Environment Not Activating
Make sure you're in the correct directory where you created the virtual environment, and double-check the activation command for your operating system.

---

**Note:** Always make sure your virtual environment is activated (you see `(orenda)` in your prompt) before installing packages or running the project.

**Dev:** Keep requirements.txt udpated
```bash 
pip list --format=freeze > requirements.txt
```