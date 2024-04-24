
# IB_XAI
Code for ''An Information Bottleneck Characterization of the Understanding-Workload Tradeoff in Human-Centered Explainable AI'' by Lindsay Sanneman*, Mycal Tucker*, and Julie Shah at FaCCT 2024.

In the paper, we explore how tradeoffs between complexity and informativeness of explanations regulate human workload and understanding of explanations.
This code provides an implementation of how to generate abstractions for such explanations.

## Setup:
### Install all the required packages

Install the packages listed under requirements.txt. We assume you're running Python 3.8.

### Adding the color subrepo:
We use the WCS dataset for our color domain experiments. You can load such data, and associated models, from an existing github repo that has done work in this area:

> git submodule add --force https://github.com/nogazs/ib-color-naming ib_color_naming

Be sure to cite Noga Zaslavsky's associated work (described in that repo).

## Running the code!

We assume all code is run from this repo's directory with the relevant virtual environment activated. For example,


> cd ~/userXXX/src/explanation_abstractions
> 
> conda activate explanations
> 
> python src/scripts/exact_grid_ib.py

This moves into the repo, activates the environment, and then runs the ``exact_grid_ib.py`` script to generate grid abstractions

The output of running ``exact_grid_ib.py`` is a series of files, located under ``saved_data/exact_ib_grid``. Each subdirectory within that will contain abstractions for each grid, for different reward functions during IB and evaluation processes.

If you want to generate abstractions for the color domain instead of the grid domain, simply run ``src/scripts/exact_color_ib.py``.

## Followup
Please contact Lindsay or Mycal if you have further questions about this work.

## Citing
If you find the paper or this codebase useful, please cite the FaCCT paper:

```
  @inproceedings{
    sannemantuckerIBXAI,
    title={An Information Bottleneck Characterization of the Understanding-Workload Tradeoff in Human-Centered Explainable AI},
    author={Lindsay Sanneman* and Mycal Tucker* and Julie Shah},
    booktitle={Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency},
    year={2024},
    publisher={Association for Computing Machinery}
  }
```
