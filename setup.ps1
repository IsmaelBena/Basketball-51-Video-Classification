pyenv install 3.9.5
pyenv local 3.9.5
pip install virtualenv
virtualenv basketball_classification_env
basketball_classification_env/Scripts/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
