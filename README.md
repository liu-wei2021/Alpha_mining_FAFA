# Alpha Mining FAMA

<p align="center">
  <a href="https://github.com/liu-wei2021/Alpha_mining_FAFA/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=liu-wei2021/Alpha_mining_FAFA" />
  </a>
</p>

## Features

- **Data loading & cleaning**
  Reads parquet OHLCV, normalizes column names, computes VWAP, forward/back-fills missing values, and calculates per-ticker returns.
- **Alpha generation**
  Implements 100+ hand‐crafted factor definitions in `AlphaFactory`.
- **IC matrix**
  Computes daily Spearman rank IC between each factor’s exposure and next‐period returns.
- **Clustering**
  Groups factors by their IC profiles via K-Means.
- **LLM-driven alpha-mining**
  Uses a DeepSeek/OpenAI‐compatible LLM to iteratively generate new factors, evaluate them by RankIC, and maintain a “chain of experience” per cluster.

## Requirements

- Python 3.8+
- pandas
- numpy
- scipy
- scikit-learn
- pyyaml
- langchain-openai
- openai

```bash
pip install pandas numpy scipy scikit-learn pyyaml
```

## Installation

```bash
git clone https://github.com/liu-wei2021/Alpha_mining_FAFA.git
cd Alpha_mining_FAFA

# (optional) create & activate venv
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt

# (optional) install editable package
pip install -e .
```

After editable install you get a console script `alpha-mining`.

## Configuration

Edit `configs/config.yaml`:

```yaml
data:
  input_parquet: "data/raw/market.parquet"
  outputs_dir:   "data/outputs_dir"

pipeline:
  n_clusters: 8
  random_state: 42
  function_definition: |
    # Functions and Operators from Alpha101:
    # returns(x): period-over-period return
    # ts_sum(x,n): rolling sum over n periods
    # sma(x,n): moving average of x
    # stddev(x,n): rolling standard deviation
    # correlation(x,y,n): rolling correlation
    # covariance(x,y,n): rolling covariance
    # ts_rank(x,n): rolling rank (percentile)
    # delta(x,n): difference over n lags
    # delay(x,n): lag by n periods
    # rank(x): cross-sectional rank into [0,1]
    # scale(x): scale so sum|x|=1
    # ts_argmax(x,n): argmax over n
    # ts_argmin(x,n): argmin over n
    # decay_linear(x,n): linear decay MA over n

deepseek:
  api_key:     "sk-your-deepseek-key"
  api_base:    "https://api.deepseek.com/v1"
  model:       "deepseek-reasoner"
  temperature: 0.0
```

## Usage

### Full pipeline

```bash
python main.py
```

1. Load & clean data  
2. Generate all alpha factors  
3. Compute IC matrix & write JSONs  
4. Cluster factors & write CSV/JSON  

### LLM-driven alpha-mining

After initial run produces `clustered_formulas.json` + `formula_rankic.json` in your outputs dir:

```bash
python main.py --output-dir data/outputs_dir --alpha-mine --iters 10
```

- `--output-dir`  folder with seed JSONs  
- `--alpha-mine`  enable LLM iterations  
- `--iters N`    run N rounds  

Writes per iteration:

- `experience_chains_iter{n}.json`  
- `time_series_rankic_iter{n}.json`  

### Mining-only shortcut

If you already have your seed JSONs:

```bash
python -c "from src.deepseek_agent import run; run(num_iters=5)"
```

or, after editable install:

```bash
alpha-mining --alpha-mine --iters 5
```

## Project Structure

```
├── configs/               
│   └── config.yaml         # settings
├── data/
│   ├── raw/                # input parquet
│   └── outputs_dir/        # JSON & CSV outputs
├── src/                    # code package
│   ├── config.py
│   ├── data_loader.py
│   ├── alpha_functions.py
│   ├── factor_matrix.py
│   ├── clustering.py
│   ├── data_fetch.py
│   ├── deepseek_agent.py
│   ├── utils/
│   │   ├── generate_rankics.py
│   │   ├── ts_functions.py
│   │   ├── stat_helpers.py
│   │   ├── math_helpers.py
│   │   └── clustered_formulas.py
│   └── constants/
│       └── formula_map.py
├── main.py                 # CLI orchestration
├── requirements.txt        # deps
└── setup.py                # pip package & entry-point
```


## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/xyz`
3. Commit your changes & push
4. Open a Pull Request

## Contributors

Thanks to the following people for their contributions:

- **Vincent Liu wei** (@liu-wei2021) – project setup, alpha function implementations, CI configuration
- **Jing Xi Wei** (@Jingxi-Wei) – project setup, alpha function implementations, CI configuration

Feel free to add yourself via a PR if you’d like to be listed here.

## License

This project is licensed under the MIT License.

---

**Happy Alpha mining!**

