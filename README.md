# Reactivaton drift

Analysis and plotting for manuscript "Sleep stages antagonistically modulate reactivation drift".

* electro-physiological data from the Csciscvari group (IST Austria) recorded 
by Peter Baracskay

# File structure

### main.py

All analysis steps are started from here

* uses parameters from parameter_files/
* all other parameters are either set as parser arguments or inside the 
"params" class

## /analysis_scripts

* scripts to generate main and supplementary figures

## /function files

### load_data.py
This is the first script that needs to be called to get the data.

* class LoadData: selects and pre-processes raw data (.des, .whl, .clu) according to session_name,
experiment_phase, cell_type
  * depending on type of data (SLEEP, EXPLORATION, TASK) different parts of the raw data are 
  returned/processes
  * returns object with data for subsequent analysis

### single_phase.py

Contains classes/functions to analyze ONE phase (e.g. sleep or behavior).

* class BaseMethods: methods to analyze sleep and awake data for ONE POPULATION
  * class Sleep: derived from BaseMethods --> used to analyze sleep data
  * class Exploration: derived from BaseMethods --> used to analyze exploration data
  * class Cheeseboard: used to analyze cheeseboard data

### multiple_phases.py

Contains classes/functions to analyze TWO or more phase (e.g. sleep and behavior).

### pre_processing.py

Contains classes/functions for pre-processing data (e.g. binning) . Is used by analysis_methods.py

### support_functions.py
Contains simple functions that are needed to compute things. Used by almost all functions/classes.

### ml_methods.py
Machine learning related approaches

## /parameter_files
* contains parameters for each session
* naming convention: [ANIMAL ID].py 

