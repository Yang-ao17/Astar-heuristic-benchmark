# A* Heuristic Benchmark

This is a small budgeted A* heuristic benchmark for grid maps. It compares how different heuristic functions affect both search effort and solution quality.

The planner uses 8-neighbor movement:

- cardinal moves cost `1.0`
- diagonal moves cost `sqrt(2)`
- diagonal corner-cutting through obstacles is not allowed

The benchmark evaluates:

- path validity
- path cost
- optimality gap
- expanded nodes
- runtime
- whether the search stopped because of the expansion budget

The motivation is related to LLM-based automated heuristic design. If an LLM or another search process proposes a candidate heuristic, this project shows how that heuristic can be evaluated automatically without adding any external API calls or requiring API keys.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_experiments.py
```

The script runs the benchmark on the `maze` and `trap` maps with `max_expansions=3000`, prints the results, saves `results/benchmark_results.csv`, and writes comparison figures into the `results/` folder.

## Project Structure

```text
astar-heuristic-benchmark/
  README.md
  requirements.txt
  run_experiments.py
  src/
    astar.py
    heuristics.py
    maps.py
    metrics.py
    visualization.py
  results/
  notebooks/
    03_Hands-On_A_star.ipynb
```
