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

### multiple_sessions.py

Contains classes/functions to analyze multiple sessions (e.g. different animals/days)

### pre_processing.py

Contains classes/functions for pre-processing data (e.g. binning) . Is used by analysis_methods.py

* class PreProcess: base class
  * class PreProcess: derived from PreProcess. Computes e.g. temporal spike binning or 
  constant nr. of spikes binning

### support_functions.py
Contains simple functions that are needed to compute things. Used by almost all functions/classes.

### ml_methods.py
Machine learning approaches for one or two populations:
* class MlMethodsOnePopulation
* class MlMethodsTwoPopulation

## /parameter_files
* [ANIMAL ID].py 
  * 3 dictionaries: 
     * data description dictionary: from .des file --> which index 
  corresponds to which phase of the experiment
     * data selection dictionary: where to find the data, which cell types
     * cell_type_sub_div_dic: defines where cell types are split (
     e.g. left and right hemisphere cells) 
* standard_analysis_parameters.py: stores all standard parameters/
place holders that are set at a later stage --> almost all later computations
use parameters stored in this class
