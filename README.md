# PlaceMentor AI

PlaceMentor AI is a Streamlit chatbot for student placement preparation. It
answers questions about aptitude, coding and data structures, interviews,
resumes, and the placement process by retrieving the closest answer from a
curated FAQ dataset. Answers are grounded in the FAQ; the application does not
generate new answers.

## Technologies

- Python 3.14
- Streamlit for the interactive chat interface
- pandas for FAQ data loading and validation
- sentence-transformers and the pretrained
  `sentence-transformers/all-MiniLM-L6-v2` model for semantic embeddings
- NumPy and scikit-learn for cosine-similarity ranking

The model runs locally on CPU. On first use, it is downloaded from Hugging Face
and FAQ question embeddings are computed. An internet connection is needed for
that initial model download.

## Project files

- `app.py` — Streamlit application entry point
- `src/placement_mentor/` — text preprocessing and FAQ retrieval
- `data/placement_faq.csv` — FAQ questions and answers used by the application
- `data/evaluation.csv` — examples for optional retrieval evaluation
- `tests/` — retrieval, application, and evaluation checks

## Installation

From the project directory, create and activate a virtual environment in
PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, invoke the virtual environment's
Python directly as shown below instead.

## Run the application

From the project directory in PowerShell:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.fileWatcherType none
```

Open the local URL printed by Streamlit in a browser. Choose a placement
category and similarity threshold in the sidebar, then enter a question or
select a suggested question. The threshold is a cosine-similarity score, not a
probability. A fallback is shown when no FAQ is sufficiently relevant.

The file watcher is disabled to avoid noisy watcher errors from lazy
Transformers imports in the working Python environment. Restart Streamlit to
load code edits.

## Tests and evaluation

Run the automated unit and application tests from the project directory:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Run the focused semantic retrieval smoke checks:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe tests\test_retrieval_smoke.py
```

To evaluate the examples in `data/evaluation.csv` and print measured retrieval
results:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe tests\evaluate_retrieval.py
```
