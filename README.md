ManualChamberFLUX
=
- Easy and standardized way to calculate CO2, CH4, N2O, and CO fluxes from manual chamber measurements. 
- See the config_manualFLUX file for more details how to configure the program. After the program is configured, run ManualChamberFLUX.py to calculate the fluxes.
- This program has been developed under Manjaro Linux distro, using Anaconda distribution for Python 3.
- The user is responsible for properly formatting the data and times files for this program. See the config_manualFLUX for more details and the tests folder for examples. There are currently no data quality checks for the data file, except that it fulfills the minimum column requirements. However, some data quality checking is done for the times file but it is very rudimentary, so don't trust it too much.

Author: Mika Korkiakoski

Bug reports: mika.korkiakoski@fmi.fi

How to install
=
Navigate to the folder containing pyproject.toml.

Run the command:

Windows:

	py -m pip install .
	
Linux:

	pip install .

Alternative install
=

If you don't have Python installed you could start by installing Anaconda distribution: https://www.anaconda.com/products/distribution

If you don't want Anaconda, install Python 3 and make sure Git is installed: https://git-scm.com/download/ (for Windows)

Finally, navigate to the folder containing the requirements.txt file and run the command:

Windows:

	py -m pip install -r requirements.txt
	
Linux:

	pip install -r requirements.txt


Version history
=
v. 1.0  
First working version

v. 1.1  
Added support for ancillary variables measured by the gas analyzer system (i.e. Juusoflux)
        Headspace temperature, air pressure, RH and PAR can now be given in the data file and their means are calcualted into the result file.
        Removed 'automatic_chamber_fluxes' from the results file name.
        Fixed a crash caused by the closure length being lower than fit_minimum parameter. Now warning is given instead of the whole program crashing.
        Fixed a crash caused when data_closure was empty even though it was supposed to give only a warning.
        Added more comments into the code.
        
v. 1.2  
Added an interactive start and end time selection for the fits and new relevant parameters into the config
        file: interactive_times and interactive_save.
        The user can now decide if they want to save the interactively selected times.
        Adjusted the program functionality to take into account cases when the gas column exists in the data file, but the column does not contain any data at all.
        Moved the time averaging of the gas time series from the fluxcalc function to the main function before possible interactive time selection.
        Added more checks to the config file.
        Added some checks to the times file (start and end times, headspace temperature, and air pressure).
        Replaced CMB_Err class with sys.exit() in the exception handling.
        Added even more comments into the code. Fixed some typos.
	Available in Github!

Known issues
=
During the interactive closure time selection, if a plot is closed by pressing the "x" in the upper right corner, the program either crashes or the program uses the previous start and end times for the closed plot, messing things up. So, don't manually close the plots! The plots are closed by clicking the plot area thrice.
