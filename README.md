# Decentralized Reinforcement Learning

## Setup


```bash
# Check to see if you have the necessary tools for building OpenSpiel:
cmake --version        # Must be >= 3.12
clang++ --version      # Must be >= 7.0.0
python3-config --help

# If not, run this line to install them.
# On older Linux distros, the package might be called clang-9 or clang-10
sudo apt-get install cmake clang python3-dev

# On older Linux distros, the versions may be too old.
# E.g. on Ubuntu 18.04, there are a few extra steps:
# sudo apt-get install clang-10
# pip3 install cmake  # You might need to relogin to get the new CMake version
# export CXX=clang++-10

# Clone this depository
git clone https://github.com/AmitTsvi/decentralized_rl_multi.git 

# Recommended: Install pip dependencies and run under virtualenv.
sudo apt-get install virtualenv python3-virtualenv
virtualenv -p python3 venv

# Set the PYTHONPATH: add the following lines to the end of venv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/arl
export PYTHONPATH=$PYTHONPATH:~/arl/open_spiel
export PYTHONPATH=$PYTHONPATH:~/arl/open_spiel/build/python

source venv/bin/activate

# Install dependencies:
pip install -r ubuntu_requirements.txt

# Finally, build OpenSpiel and install its dependencies:
pip3 install --upgrade setuptools pip
cd open_spiel
./install.sh
./open_spiel/scripts/build_and_run_tests.sh

```

For GPU, set OMP_NUM_THREADS to 1: `export OMP_NUM_THREADS=1`.

## Training
Run `python runner.py --<experiment-name>` to print out example commands for the environments in the paper. Add the `--for-real` flag to run those commands. You can enable parallel data collection with the `--parallel_collect` flag. You can also specify the gpu ids. As examples, in `runner.py`, the methods that launch `bandit`, `chain`, and `duality` do not use gpu while the others use gpu 0.

## Visualization
You can view the training curves in `<exp_folder>/<seed_folder>/group_0/<env-name>_train/quantitative` and you can view visualizations (for environments that have image observations) in `<exp_folder>/<seed_folder>/group_0/<env-name>_test/qualitative`.

## Credits
The PPO update is based on [this repo](https://github.com/Khrylx/PyTorch-RL.git).
