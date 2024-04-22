# eo-master2: Eart observation

# Installation

## Package manager
the used package manager is `pipenv`

### Requierments

#### Lunix|Macos user
For Linux or WSL users, you need to install gdal in your system before installing python envirenment

```bash
sudo apt install -y gcc curl libmariadb-dev libpq-dev git iputils-ping python3-pip libmysqlclient-dev libgdal-dev=3.4.1
```
#### Windows users
the wheel can be downloaded from: https://github.com/cgohlke/geospatial-wheels/releases.

The wheel is donwloaded and is in this repository. 
You have the comment/uncomment the line of gdal package in the `Pipefile` according to the operating system.

#### Env installation
After this step, the installation of the envirennement is done by running the following command

```bash
pipenv install
```


