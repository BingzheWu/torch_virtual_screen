pip install --upgrade pip
conda update -n base conda
conda config --add channels pytorch
conda config --add channels rdkit
conda config --add channels conda-forge
conda config --add channels rmg
conda install rdkit=2019.03.4.0=py36hc20afe1_1
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c dglteam dgllife dgl-cu10.0


