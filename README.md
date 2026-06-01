# Pacman

## Environment setup

This project uses [`uv`](https://docs.astral.sh/uv/) to manage Python and
dependencies. The project Python version is pinned in `.python-version`, and
dependencies are locked in `uv.lock`.

### 1. Install uv

macOS or Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installing, restart your terminal if the `uv` command is not found.

### 2. Install the project environment

From the repository root:

```sh
uv sync
```

`uv sync` creates a local `.venv` directory and installs the exact dependency
versions from `uv.lock`. If Python 3.14 is not already installed, uv can
download it automatically. To install it explicitly:

```sh
uv python install 3.14
```

### 3. Run project commands

Run commands inside the environment with `uv run`, for example:

```sh
uv run python main.py
```

Or activate the virtual environment first:

macOS or Linux:

```sh
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
```
