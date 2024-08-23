# FATES land use data processing tool

A python tool for processing LUH2 data for use with FATES landuse run modes

## Purpose

This tool creates the necessary land use datasets to be used as inputs for running FATES with land use cover and change.  The tool has two major options, i.e. python subcommands, to provide the data set of interest:

- land use timeseries data
- land use association to plant functional type (land use x pft) static mapping data

The tools provides these data sets as netcdf files by concatenating the various raw input data sets into a single file.  The tool provides the ability to regrid the source data resolution to a target
resolution that the user designates for either output option.  The output data is then usable by FATES, mediated through a host land model (currently either CTSM or E3SM).

## Input data sources

The tool requires input data from two primary sources: the [Land Use Harmonization](https://luh.umd.edu/), or LUH2, historical data sets and the [THESIS Tools Datasetes](https://doi.org/10.5065/29s7-7b41).  A static mapping file from the LUH2 data is required for both output options.  The historical LUH2 transient, state, and management data is used as input for the FATES land use timeseries data output option, whereas the THEMIS tools forest, pasture, "other" and current surface data sets are necessary for the land use x pft static map output option.  These data sets can be downloaded directly from their respective sources and are not provided as part of this package.

### LUH2 Historical data sets

Generating the land use timeseries dataset requires as input the historical datasets from the [LUH2 v2h Release](https://luh.umd.edu/data.shtml):

_Historical Files_
- [states](https://luh.umd.edu/LUH2/LUH2_v2h/states.nc)
- [transitions](https://luh.umd.edu/LUH2/LUH2_v2h/transitions.nc)
- [management](https://luh.umd.edu/LUH2/LUH2_v2h/management.nc)

_Supporting Files_
- [static data](https://luh.umd.edu/LUH2/LUH2_v2h/staticData_quarterdeg.nc)

The documenatation for the historical datasets is [available for download as a PDF](https://luh.umd.edu/LUH2/LUH2_v2h_README.pdf).

### CLM5 THESIS datasets

Generating the land use x pft static mapping dataset requires as input the CLM 5 land use data tool 1/4 degree ouput datasets which can be downloaded via the [UCAR Geoscience Data Exchange](https://gdex.ucar.edu/dataset/188b_oleson/file.html):

- [Current forest](https://gdex.ucar.edu/dataset/188b_oleson/file/CLM5_current_luhforest_deg025.nc)
- [Current pasture](https://gdex.ucar.edu/dataset/188b_oleson/file/CLM5_current_luhpasture_deg025.nc)
- [Current other](https://gdex.ucar.edu/dataset/188b_oleson/file/CLM5_current_luhother_deg025.nc)
- [Current 1/4 deg surface](https://gdex.ucar.edu/dataset/188b_oleson/file/CLM5_current_surf_deg025.nc)

The LUH2 static data file noted in the LUH2 historical dataset section above is also required as an input.

## Installation

### Dependencies

This python package depends upon the following packages (which can be found in the pyproject.toml): 

- [`xesmf`](https://pangeo-xesmf.readthedocs.io/en/latest/index.html#): provides regridding capability
- `netcdf4`: requirement for writing `.nc` files and necessary for `xesmf` dependencies 

Note that `xesmf` relies on `esmpy` which is currently not available via PyPi (and thus not `pip` installabele).  See `esmf` issue [#256](https://github.com/esmf-org/esmf/issues/256) for discussion on this subject.  The [`xesmf` installation documentation](https://pangeo-xesmf.readthedocs.io/en/latest/installation.html#notes-about-esmpy) also provides some discussion of the issue.

### Conda install

This package is available through the [ngeetropics anaconda channel](https://anaconda.org/ngeetropics/tools-fates-landusedata).  Note that the package is not available through PyPi as the `xesmf` dependency is only available through conda.

To install in an existing environment:
``` sh
conda install ngeetropics::tools-fates-landusedata
```

To install the tool in a new environment:

``` sh
conda create -n <new-env-name> ngeetropics::tools-fates-landusedata
```

## Usage

### From source

It is possible to use this tool without installing it using conda by obtaining the source files.  Note that this assumes that a conda environment has been created with the necessary dependencies as listed in the [`pyproject.toml`](https://github.com/NGEET/tools-fates-landusedata/pull/14) manifest.  The source code for this repository is available as a submodule of the fates repository and can be found in the `fates/tools` directory.  To run the tool from source with `conda run`:

Change directory to `fates/tools/landusedata/src/` and run:

``` sh
conda run -n <conda-env> python -m landusedata <subcommand> <positional-args>
```

### Via Conda

This tool is meant to be utilized from the command line.  The top level call for the command line with help is shown below:
``` sh
$ fates-landusedata -h
usage: fates-landusedata [-h] {luh2,lupft} ...

FATES landuse data tool

options:
  -h, --help    show this help message and exit

subcommands:
  {luh2,lupft}  landuse data tool subcommand options
    luh2        generate landuse harmonization timeseries data output
    lupft       generate landuse x pft static data map output
```

The `luh2` subcommand call tells the tool to build the combined landuse timeseries data:

``` sh
$ fates-landusedata luh2 -h
usage: luh2 [-h] [-w REGRIDDER_WEIGHTS] [-b BEGIN] [-e END] [-o OUTPUT] regrid_target_file luh2_static_file luh2_states_file luh2_transitions_file luh2_management_file

positional arguments:
  regrid_target_file    target surface data file with desired grid resolution
  luh2_static_file      luh2 static data file
  luh2_states_file      full path of luh2 raw states file
  luh2_transitions_file
                        full path of luh2 raw transitions file
  luh2_management_file  full path of luh2 raw management file

options:
  -h, --help            show this help message and exit
  -w REGRIDDER_WEIGHTS, --regridder_weights REGRIDDER_WEIGHTS
                        filename of regridder weights to save
  -b BEGIN, --begin BEGIN
                        beginning of date range of interest
  -e END, --end END     ending of date range to slice
  -o OUTPUT, --output OUTPUT
                        output filename
```

The `lupft` subcommand tells the tool to build the landuse x pft mapping file:

``` sh
$ fates-landusedata lupft -h
usage: lupft [-h] [-o OUTPUT] regrid_target_file luh2_static_file clm_luhforest_file clm_luhpasture_file clm_luhother_file clm_surface_file

positional arguments:
  regrid_target_file    target surface data file with desired grid resolution
  luh2_static_file      luh2 static data file
  clm_luhforest_file    CLM5_current_luhforest_deg025.nc
  clm_luhpasture_file   CLM5_current_luhpasture_deg025.nc
  clm_luhother_file     CLM5_current_luhother_deg025.nc
  clm_surface_file      CLM5_current_surf_deg025.nc

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output filename

```
