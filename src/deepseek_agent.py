# src/deepseek_agent.py

import json
from pathlib import Path
import os
import re

import pandas as pd
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import openai

from .config import settings
from .data_loader import load_factor_df
from .utils.generate_rankics import factor_series_fn, compute_rankic

# Wire your OpenAI key
openai.api_key = settings.deepseek.api_key
openai.api_base = settings.deepseek.api_base  # <— point at DeepSeek’s endpoint

# Initialize LangChain OpenAI client
llm = ChatOpenAI(
    model_name=settings.deepseek.model,
    temperature=settings.deepseek.temperature,
    openai_api_key=settings.deepseek.api_key,
    openai_api_base=settings.deepseek.api_base,
)

# Prompt template for factor generation
prompt_template = """
Instruction
You are an alpha generator. You should follow the following rules:
1. The inputs are the alpha factors that are currently performing well, and you are
   required to output a new alpha factor that is generated from the fusion of
   these factors, and your factor must be different from the input factor.
2. Do not repeat example answer.
3. You should return new different factors in a json array.
4. The specific function is defined as follows:
{function_definition}
5. Follow the path in "improve_path". -> Indicates that the following factors have
   better performance than the previous factors. You should refer it to build new
   alpha.

Input Example
alphas: {css}
generate_factor_num: 1
improve_path: {chain}

Output Example
["rank(correlation(open, volume, 10) / rank(open))"]
"""


def get_latest_iteration(prefix: str, out_dir: Path) -> int:
    """Scan out_dir for files named prefix_iter{n}.json and return max n or 0."""
    matches = [p.stem for p in out_dir.iterdir() if p.stem.startswith(prefix)]
    iters = [int(name.split("_iter")[-1]) for name in matches if "_iter" in name]
    return max(iters) if iters else 0


def one_iteration(
    df: pd.DataFrame,
    chains: dict,
    rankic: dict,
    out_dir: Path,
    iteration: int
):
    new_chains = {}
    new_rankic = rankic.copy()

    for cid, chain in chains.items():
        prompt = prompt_template.format(
            function_definition=settings.pipeline.function_definition,
            css=json.dumps(chain, ensure_ascii=False),
            chain=" -> ".join(chain)
        )

        resp = llm.invoke([
            SystemMessage(content=(
                "You are an alpha-mining agent implementing the FActor Mining Agent (FAMA) framework. "
                "Generate a new, interpretable financial alpha factor expression as JSON."
            )),
            HumanMessage(content=prompt)
        ])

        raw = resp.content
        # ── strip Markdown code fences (```json … ```) if present ──
        fence = re.match(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        if fence:
            raw = fence.group(1).strip()
        print(f"[Cluster {cid}] LLM output:\n{raw}")

        try:
            gen = json.loads(raw)
            factor = gen[0] if isinstance(gen, list) else gen
        except json.JSONDecodeError:
            print(f"[Cluster {cid}] ❌ JSON parse error, skipping.")
            new_chains[cid] = chain
            continue

        try:
            series = factor_series_fn(df, factor)
            ric = compute_rankic(series, df["return_"])
        except Exception as e:
            print(f"[Cluster {cid}] ❌ eval error: {e}")
            new_chains[cid] = chain
            continue

        new_rankic[factor] = ric
        updated = chain + [factor]
        # keep top-15 by |RankIC|
        kept = sorted(updated, key=lambda f: abs(new_rankic.get(f, 0)))[-15:]
        new_chains[cid] = kept
        print(f"[Cluster {cid}] ✅ {factor} → RIC={ric:.4f}")

    # Save iteration JSONs
    chains_file = out_dir / f"experience_chains_iter{iteration}.json"
    rankic_file = out_dir / f"time_series_rankic_iter{iteration}.json"
    chains_file.write_text(json.dumps(new_chains, indent=2, ensure_ascii=False), encoding='utf-8')
    rankic_file.write_text(json.dumps(new_rankic, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"✅ Saved iteration {iteration}: {chains_file}, {rankic_file}")


def run(num_iters: int = 50):
    """Driver loop to run multiple DeepSeek alpha-mining iterations."""
    df = load_factor_df()
    out_dir = Path(settings.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(num_iters):
        last = get_latest_iteration("experience_chains", out_dir)
        if last == 0:
            chain_file = out_dir / "clustered_formulas.json"
            rankic_file = out_dir / "formula_rankic.json"
        else:
            chain_file = out_dir / f"experience_chains_iter{last}.json"
            rankic_file = out_dir / f"time_series_rankic_iter{last}.json"

        chains = json.loads(chain_file.read_text(encoding='utf-8'))
        rankic = json.loads(rankic_file.read_text(encoding='utf-8'))

        one_iteration(df, chains, rankic, out_dir, last + 1)
