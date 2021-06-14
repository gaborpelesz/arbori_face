# Install

For installation you need do download the latest version of python (e.g.: 3.8.1).  
https://www.python.org/downloads/

### Clone repository
```bash
git clone https://<LOGIN>@bitbucket.org/exahexa/boriface.git

cd boriface
```

### Install pipenv

> Recommended: Homebrew / Linuxbrew
```bash
brew install pipenv
```

> Alternative way
```bash
pip3 install --user pipenv
```

### Get into pipenv shell (you need to be in the 'boriface' directory)
```bash
pipenv shell
```

### Install dependencies through pipenv
```bash
pipenv install
```
*(this step may take a few minutes to finish)*

# Running

### Get into pipenv shell (if not already in)
*(assuming that the current directory is 'boriface')*
```bash
pipenv shell
```

### Run program
```bash
python main.py
```

First start up may take up a few minutes and **requires** internet connection, because the models doing the Face detection and Face recognition will be downloaded.