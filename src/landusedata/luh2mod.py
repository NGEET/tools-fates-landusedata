import re, sys
import numpy as np
import xarray as xr



# Import luh2 or surface data sets
def ImportLUH2TimeSeries(input_file, start=None, stop=None):
    # Open files
    # Set decode_times to false as the luh2 raw data is outside the range
    # of the standard NetCDF datetime format.
    datasetout = xr.open_dataset(input_file, cache=False, decode_times=False)
    print("Input file dataset opened: {}".format(input_file))

    # Prep the input data for use
    datasetout = PrepDataset(datasetout, start, stop)

    return datasetout


# Prepare the input_file to be used for regridding
def PrepDataset(input_dataset, start=None, stop=None):
    # Use the maximum span if start and stop are not present
    # Note that while both LUH2 and LUH3 data have annual time steps,
    # the LUH2 data has units of 'years since 850-01-01' while the LUH3 data
    # has units of 'days since 850-01-01'.

    # Get the units to determine the file time
    # It is expected that the units of time is 'years since ...'
    # Convert days to years if necessary
    orig_unit = input_dataset.time.units
    if orig_unit.startswith("days"):
        input_dataset["time"] = input_dataset.time.values // 365
        input_dataset.time.attrs["units"] = orig_unit.replace("days", "years")

    # Check that the time units are correct, otherwise abort
    time_since_array = input_dataset.time.units.split()
    if not (time_since_array[0] == "years"):
        sys.exit(
            "FileTimeUnitsError: input file units of time is not 'years since ...' or 'days since ...'"
        )

    # Note that datetime package is not used as the date range might
    # be beyond the bounds of the packages applicable bounds
    time_since = int(time_since_array[2].split("-")[0])

    # Get the time bounds of the input file
    start_bound = input_dataset.time.values[0]
    stop_bound = input_dataset.time.values[-1]

    # If no input provided, simply get the bounds of the time
    if isinstance(start, type(None)):
        start = start_bound + time_since

    if isinstance(stop, type(None)):
        stop = stop_bound + time_since

    # Convert the input dates to years since 0850
    years_since_start = start - time_since
    years_since_stop = stop - time_since

    # Abort if the times provided are outside the applicable range
    if (
        years_since_start < start_bound
        or years_since_stop < start_bound
        or years_since_start > stop_bound
        or years_since_stop > stop_bound
    ):
        sys.exit(
            "StartStopBoundError: the input start or stop date is outside the applicable range of {} to {}".format(
                time_since + start_bound, time_since + stop_bound
            )
        )

    # Truncate the data to the user defined range
    # This might need some more error handling for when
    # the start/stop is out of range
    if time_since_array[0] == "years":
        input_dataset = input_dataset.sel(
            time=slice(years_since_start, years_since_stop)
        )
    else:
        input_dataset = input_dataset.sel(
            time=slice(years_since_start * 365.0, years_since_stop * 365.0)
        )

    # Save the timesince as a variable for future use
    input_dataset["timesince"] = time_since


    return input_dataset



def make_evenly_distributed_bounds_array(orig_array):
    step = (orig_array[1] - orig_array[0]) / 2.0
    lat_b = np.append((orig_array - step), orig_array[-1] + step)
    # print(lat_b)
    # print(len(lat_b))
    return lat_b


# Temporary: Add minor correction factor to assure states sum to one
def CorrectStateSum(input_dataset):
    # Only calculate the state sum to unity correction for the appropiate dataset
    # TO DO: Update this to use the check function
    if not (
        any("irrig" in var for var in input_dataset)
        or any("_to_" in var for var in input_dataset)
    ):
        # Drop the secma and secmb variables temporarily
        temp_dataset = input_dataset.drop({"secma", "secmb"})

        # Sum the remaining state variables and normalize
        state_sum = temp_dataset.to_array().sum(dim="variable")
        state_sum = state_sum.where(state_sum != 0)
        temp_dataset = temp_dataset / state_sum

        # Update dataset with new scaled values
        input_dataset.update(temp_dataset)

        # Save the correction value
        input_dataset["stscf"] = 1.0 / state_sum

    return input_dataset
