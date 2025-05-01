# Alpha Mining FAFA

A pipeline for computing and clustering information‐coefficient (IC) matrices of quantitative “Alpha” factors on financial time series.

## Features

- **Data loading & cleaning**: reads parquet data, normalizes columns, forward/backfills missing values
- **Alpha generation**: implements 100+ factor definitions via `AlphaFactory`
- **IC matrix**: computes daily Spearman IC between factor exposures and next‐period returns
- **Clustering**: groups factors by their IC profiles using K‐Means

## Requirements

- Python 3.8+
- pandas
- numpy
- scipy
- scikit-learn
- pyyaml

```bash
pip install pandas numpy scipy scikit-learn pyyaml
```

## Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/liu-wei2021/Alpha_mining_FAFA.git
   cd Alpha_mining_FAFA
   ```
2. (Optional) Create & activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Settings are loaded by `config.py` from `configs/config.yaml` citeturn3file0. The YAML file defines:

```yaml
data:
  # Path to the input OHLCV parquet file
  input_parquet: path/to/market_data.parquet

pipeline:
  # Number of clusters for KMeans clustering
  n_clusters: 5
  # Random seed for reproducibility
  random_state: 42
```

- `settings.input_path` maps to `data.input_parquet`.  
- `settings.n_clusters` maps to `pipeline.n_clusters`.  
- `settings.random_state` maps to `pipeline.random_state`.  

## Usage

Run the pipeline end‑to‑end:

```bash
python main.py
```

This will:
1. Load & clean your data
2. Generate all alpha factors per ticker
3. Compute the daily IC matrix
4. Cluster factors and log cluster assignments

## Project Structure

```
├── .pytest_cache/         # pytest cache directory
├── .vscode/               # VSCode settings
├── configs/               # pipeline & data settings
│   └── config.yaml
├── data/                  # raw and input data files
├── outputs_dir/           # generated outputs (figures, tables)
├── src/                   # source code
│   ├── config.py          # loads YAML settings
│   ├── data_loader.py     # load & clean data
│   ├── alpha_functions.py # alpha factor definitions
│   ├── factor_matrix.py   # compute IC matrix
│   ├── clustering.py      # cluster factors by IC profiles
│   ├── ts_functions.py    # time-series utility functions
│   ├── stat_helpers.py    # rolling stats & ranking functions
│   ├── math_helpers.py    # miscellaneous math utilities
│   └── main.py            # pipeline orchestration
├── tests/                 # unit tests
│   └── ...
├── API Key                # API credentials (gitignored)
├── conftest.py            # pytest fixtures
├── LICENSE                # project license
├── README.md              # project documentation
├── requirements.txt       # Python dependencies
└── pytest.ini             # pytest configuration
```

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/xyz`
3. Commit your changes & push
4. Open a Pull Request

## License

This project is licensed under the MIT License.

---

**Happy Alpha mining!**

