# Source from SLURM scripts / interactive shells: modules + venv for I2P-Flow on the Alliance cluster.
# No python module: the venv uses uv's standalone CPython 3.8 (see scripts/setup_env.sh).
# gcc 9.3 + nvcc 11.7 are only needed to build GeoTransformer's extension.
module load StdEnv/2020 gcc/9.3.0 cuda/11.7 >/dev/null 2>&1
source ${I2P_ENV:-/project/6101778/masibo/uv_envs/cofii2p}/bin/activate
export CUDA_HOME=$EBROOTCUDA PIP_CONFIG_FILE=/dev/null
