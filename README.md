# piptools-demo

Demo project to test pip-tools with multiple libraries.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install pip-tools
pip-compile requirements.in
pip-compile dev-requirements.in

pip install -r requirements.txt -r dev-requirements.txt
