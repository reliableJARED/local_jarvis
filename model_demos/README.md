# Project Setup

Set up the virtual environment for Orenda.

## Prerequisites

- Python 3.6 or higher installed on your system
- Git (for cloning the repository)
- Make sure you're in the model_demos directory
```cmd
   cd model_demos
```

## Virtual Environment Setup

### Windows

1. **Create the virtual environment:**
   ```cmd
   python -m venv orenda
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
   python3 -m venv orenda
   ```

2. **Activate the virtual environment:**
   ```bash
   source orenda/bin/activate
   ```

3. **Verify activation:**
   You should see `(orenda)` at the beginning of your terminal prompt.

## Installing Dependencies

Once your virtual environment is activated, install the required packages:

```bash
python dependency_manager.py
```

## Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

### Python Command Not Found (Windows)
If `python` is not recognized, try using `py` instead:
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