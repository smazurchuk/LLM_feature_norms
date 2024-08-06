# LLM_feature_norms

This repository corresponds to an abstract submitted to CCN 2024

There are 3 python scripts and this is designed to be run on a computer with multiple GPUs.

The main steps are are:
 
 1. For a given feature, generate comparisons across words. This is done in `compute_LLM_comparisons.py`
 2. Run the features in parallel, using as many GPUs a machine has. The script `runLLM_parallel.py` runs the previous script in parallel
 3. Process outputs and generate figures for poster. The final script is `Generate_poster_figures.py`

 **To understand/see the prompts provided to the LLM, see `compute_LLM_comparisons.py`**