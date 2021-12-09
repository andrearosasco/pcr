# Set up environment
To set up the environment, paste the following command in a prompt:
- mamba create -n robotologyenv
- mamba activate robotologyenv
- mamba install -c conda-forge -c robotology yarp gazebo icub-main gazebo-yarp-plugins icub-models
- mamba install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
- mamba install -c defaults intel-openmp -f
- mamba install scipy
- mamba install -c conda-forge torchmetrics==0.5.1
- mamba install -c conda-forge pytorch-lightning
- mamba install -c conda-forge wandb
- mamba install -c conda-forge matplotlib
- mamba install -c conda-forge timm

NOTE: Gazebo can be started ONLY by the Windows Prompt, not by the PowerShell (the default bash in PyCharm)!