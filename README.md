
---

## Overview

This application is a **Text Summarization Web App** built using **Streamlit** and **Transformer models** (T5, BART, Pegasus).
It supports both **Abstractive** summarization (AI generates new text) and **Extractive** summarization (picks best original sentences).

This version includes:

* Legacy T5 mode (old summarizer style output)
* CPU-safe pipeline (Windows friendly)
* Length control (Short / Medium / Long)
* Multiple model support

---

## Features

| Feature                   | Description                                         |
| ------------------------- | --------------------------------------------------- |
| Abstractive Summarization | Uses T5 / BART / Pegasus transformer models         |
| Extractive Summarization  | Based on sentence scoring & frequency ranking       |
| Legacy T5 Mode            | Matches older T5 summarization output style exactly |
| Model Select              | Choose between T5-small / T5-base / BART / Pegasus  |
| Length Control            | Short / Medium / Long summary presets               |
| Download Summary          | Export as .txt                                      |

---

## Requirements

### Prerequisites

* Python 3.10 or higher recommended

### Install dependencies

```bash
pip install -r requirements.txt
```

> If you face PyTorch GPU/meta error on Windows, install CPU version:

```bash
pip uninstall -y torch
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

---

## How to Run Locally

```bash
# 1) Create venv
python -m venv .venv

# 2) Activate venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell

# 3) Install packages
pip install -r requirements.txt

# 4) Run app
streamlit run summarization.py
```

Then open browser at:
**[http://localhost:8501](http://localhost:8501)**

---

## File Structure

```
text_summarization/
│
├── summarization.py        # main Streamlit application file
├── requirements.txt        # dependencies
└── README.md               # documentation
```

---

## Notes

* Internet required only on **first run** (model auto downloads).
* For best compatibility use **t5-small** first.
* Legacy T5 Mode checkbox will produce output very similar to old scripts.

---

## License

This project is open-source and available under the **MIT License**.

---

