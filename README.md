# Can We Reverse In-Context Knowledge Edits?

## Environment
We use the [Mamba](https://github.com/mamba-org/mamba) package manager.
Install [Miniforge](https://github.com/conda-forge/miniforge)

```sh
mamba env create -f env.yml
conda activate reed
```

## Detecting and Reversing In-Context Edits
Code for detection can be found  under [./notebooks/detection.ipynb](./notebooks/detection.ipynb)

Code for training reversal tokens (RTs) can be found  under [./src/prompt_tuning.py](./src/prompt_tuning.py)

Example for continuous RTs: `python prompt_tuning.py --model meta-llama/Meta-Llama-3.1-8B --seed 1 --epochs 3  --rt 1 --lweight 0 --rf 1 --dataset counterfact`

For discrete RTs, you have to do the following: 
* To learn context dimensions, run: `python find_context_dims.py --model meta-llama/Meta-Llama-3.1-8B --seed 1 --epochs 3  --rt 1 --lweight 0.5 --rf 1 --dataset counterfact`
* To postprocess context dimensions, run: [./notebooks/find_context_dims.ipynb](./notebooks/find_context_dims.ipynb) for the corresponding model
* For learning discrete KEs, run: `python prompt_tuning.py --model meta-llama/Meta-Llama-3.1-8B --seed 1 --epochs 3  --rt 1 --lweight 0.5 --rf 1 --discrete --context_weights --dataset counterfact`

## Acknowledgement 
The code in this repository is based on  "Can We Edit Factual Knowledge by In-Context Learning?" Zheng et al., 2023


