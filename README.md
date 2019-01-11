# alexmods
Basic python modules for science analysis by Alex Ji

### Citations and references
If you use any of these tools in published work (especially data tables),
PLEASE get in contact with me about what you should cite for the parts you use.
I will compile these eventually, but it will take some time.

### Installation:

Currently recommended to install from source.
Go to a directory where you want to keep the code. Then:
```
git clone https://github.com/alexji/alexmods.git
cd alexmods
python setup.py install
```

If you expect to change any code, instead do `python setup.py develop`

`gaiatools` requires [pyia](https://github.com/adrn/pyia), [gala](https://github.com/adrn/gala), [galpy](https://github.com/jobovy/galpy), and [gaia_tools](https://github.com/jobovy/gaia_tools). The last one needs an environment variable set.

### To do list
- [ ] Make installable (seems not to work all the time)
