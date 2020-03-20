# Notations on the Book

## Chapter 2
* Hardware and Software Requirements
    * TorchVision 0.4.1 contains a bug that breaks with Pillow 7.0.0
      (see [GitHub Issue](https://github.com/pytorch/vision/issues/1712)). This can be fixed by pinning Pillow to version 6.2.1.

## Chapter 4
* The Cross-Entropy Method on FrozenLake
    * In the list of possible improvements (introducing file `Chapter04/03_frozenlake_tweaked.py`) it is not mentioned that the
      percentile should also be changed (e.g. from `70` to `30`).
