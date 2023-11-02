# rBLAJK
In silico verification of Riboswitch design for Coxsackievirus detection

# Environment setup:
Start with trying: "conda env create -f nupack_env_draw.yml" . I had some problems with the arm64 wheel of nupack 4.0.1.8 so I have been using version 4.0.0.23 instead. If building the environment from the yml file does not work, you should be able to build the environment using this series of commands:

conda create --name test_nupack_env python=3.9
conda activate test_nupack_env
conda install -c beliveau-lab nupack
pip install draw_rna
