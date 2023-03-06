# XRLFLOW

XRLFLOW is a GNN based neural network optimisation extension of [TASO](https://github.com/jiazhihao/TASO).
The basic idea was to substitute TASO's *cost-based backtracking search*
and replace it with a RL-based optimisation. Potentially, this should be
able to generalise to large unseen graphs and find better performing solutions
than the backtracking search.

XRLFLOW consists of three main parts:

1. An extension of the TASO library exposing a low-level RL environment
2. A gym-style high-level environment written in Python, interacting with the XRLFLOW environment
3. A PPO agent interacting with this environment, iteratively applying graph substitutions to a target graph.

## Setup

XRLFLOW interacts with TASO and thus depends on the TASO library. TASO on the other
hand depends on a working CUDA and CuDNN installation.

### Installing Python

The experiments used **Python 3.9**, though later versions might also work. It is **strongly**
advised to use a virtual environment such as [pyenv](https://github.com/pyenv/pyenv-installer)
or venv for installation.

### Installing CUDA

At the time of the initial experiments, we used **Cuda 10.2** and **CuDNN 7.6.5.32-1**.

It is **recommanded** to use a nvidia docker

### Installing TASO

Build taso in the "taso-build" directory, follow its instructions

Most importantly, set the `TASO_HOME` variable:

```bash
export TASO_HOME=/path/to/taso
```

Also, for some builds to succeed I had to set the `LD_LIBRARY_PATH` variable:

```bash
export LD_LIRBARY_PATH=/usr/local/lib
```

Then follow TASO's installation page.

Note that, **check installation** by 'import taso' after built.

### Installing XRLFLOW

The installation of XRLFLOW is very similar to that of TASO. First, make sure the environment
variables are set:

```bash
export TASO_HOME=/path/to/taso
export LD_LIRBARY_PATH=/usr/local/lib
```

Then, go to `taso_ext`, create a `build` folder, and run `cmake`:

```bash
cd taso_ext
mkdir build
cd build
cmake ..
make -j
sudo make install
```

At this path, you should now have two `.so` files:

```plain
ls -alp /usr/local/lib/
# ...
# -rw-r--r--  1 root root   448920 Apr  7 13:16 libtaso_rl.so
# -rw-r--r--  1 root root  1173704 Apr  7 13:39 libtaso_runtime.so
# ...
```

The extension has been built. Now the python package can be installed:

```
# In the project root:
pip install -e python/
```
