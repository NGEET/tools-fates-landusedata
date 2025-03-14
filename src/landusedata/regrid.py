import numpy as np
import xesmf as xe
import xarray as xr

import sys

from .utils import ImportRegridTarget

class TwoStepRegridder:

    def __init__(self, ds_to_regrid, regridder_steptwo_weights, intermediate_regridding_file, regrid_method="conservative"):
        
        self.se_regridder = make_se_regridder(regridder_steptwo_weights, regrid_method=regrid_method)
        intermediate_ds = ImportRegridTarget(intermediate_regridding_file)
        #print(intermediate_ds)
        self.step_one_regridder = xe.Regridder(ds_to_regrid, intermediate_ds, regrid_method)
        #print(self.step_one_regridder)

    def two_step_regridding(self, ds_to_regrid):
        ds_intermediate = self.step_one_regridder(ds_to_regrid)
        ds_se = self.se_regridder(ds_intermediate).squeeze("lat", drop=True)
        #print(ds_se)
        return ds_se.rename({"lon":"lndgrid"}).drop_vars("lndgrid")


def RegridConservative(ds_to_regrid, ds_regrid_target, regridder_weights, regrid_reuse, regrid_method="conservative", intermediate_regridding_file=None):

    # define the regridder transformation
    regridder = GenerateRegridder(ds_to_regrid, ds_regrid_target, regridder_weights, regrid_reuse, regrid_method=regrid_method, intermediate_regridding_file=intermediate_regridding_file)

    # Loop through the variables to regrid
    ds_regrid = RegridLoop(ds_to_regrid, regridder)

    return (ds_regrid, regridder)

def GenerateRegridder(ds_to_regrid, ds_regrid_target, regridder_weights_file, regrid_reuse, regrid_method = "conservative", intermediate_regridding_file=None):
    
    print("\nDefining regridder, method: ", regrid_method)
    #print(ds_to_regrid.dims)
    if 'lat' not in ds_regrid_target.dims:
        two_step_regridder = TwoStepRegridder(ds_to_regrid, regridder_weights_file, intermediate_regridding_file, regrid_method=regrid_method)
        regridder = two_step_regridder.two_step_regridding #make_se_regridder(regridder_weights_file, regrid_method=regrid_method)

    elif (regrid_reuse):
        regridder = xe.Regridder(ds_to_regrid, ds_regrid_target,
                                 regrid_method, weights=regridder_weights_file)
    else:
        regridder = xe.Regridder(ds_to_regrid, ds_regrid_target, regrid_method)

        # If we are not reusing the regridder weights file, then save the regridder
        filename = regridder.to_netcdf(regridder_weights_file)
        print("regridder saved to file: ", filename)

    return(regridder)


def make_se_regridder(weight_file, regrid_method):
    weights = xr.open_dataset(weight_file)
    in_shape = weights.src_grid_dims.load().data.tolist()[::-1]

    # Since xESMF expects 2D vars, we'll insert a dummy dimension of size-1
    if len(in_shape) == 1:
        in_shape = [1, in_shape.item()]

    # output variable shape
    out_shape = weights.dst_grid_dims.load().data#.tolist()[::-1]
    if len(out_shape) == 1:
        out_shape = [1, out_shape.item()]

    #print(in_shape, out_shape)
    #print(weights)
    #print(len(weights.yc_a.data.reshape(in_shape)[:, 0]))
    #print(weights.yc_a.data.reshape(in_shape)[:, 0])
    #print(len((weights.xc_a.data.reshape(in_shape)[0, :])))
    #print((weights.xc_a.data.reshape(in_shape)[0, :]))
    #sys.exit(4)
    dummy_out = xr.Dataset(
        {
            "lat": ("lat", np.empty((out_shape[0],))),
            "lon": ("lon", np.empty((out_shape[1],))),
        }
    )
    dummy_in= xr.Dataset(
        {
            "lat": ("lat", weights.yc_a.data.reshape(in_shape)[:, 0]),
            "lon": ("lon", weights.xc_a.data.reshape(in_shape)[0, :]),
        }
    )

    regridder = xe.Regridder(
        dummy_in,
        dummy_out,
        weights=weight_file,
        #method="conservative_normed",
        #method=regrid_method,
        method="bilinear",
        reuse_weights=True,
        periodic=True,
    )
    return regridder

def RegridLoop(ds_to_regrid, regridder):

    # To Do: implement this with dask
    print("\nRegridding")

    # Loop through the variables one at a time to conserve memory
    ds_varnames = list(ds_to_regrid.variables.keys())
    varlen = len(ds_to_regrid.variables)
    first_var = False
    for i in range(varlen-1):

        # Skip time variable
        if (not "time" in ds_varnames[i]):

            # Only regrid variables that match the lat/lon shape.
            if (ds_to_regrid[ds_varnames[i]][0].shape == (ds_to_regrid.lat.shape[0], ds_to_regrid.lon.shape[0])):
                print("regridding variable {}/{}: {}".format(i+1, varlen, ds_varnames[i]))

                # For the first non-coordinate variable, copy and regrid the dataset as a whole.
                # This makes sure to correctly include the lat/lon in the regridding.
                if (not(first_var)):
                    ds_regrid = ds_to_regrid[ds_varnames[i]].to_dataset() # convert data array to dataset
                    ds_regrid = regridder(ds_regrid)
                    first_var = True

                # Once the first variable has been included, then we can regrid by variable
                else:
                    ds_regrid[ds_varnames[i]] = regridder(ds_to_regrid[ds_varnames[i]])
            else:
                print("skipping variable {}/{}: {}".format(i+1, varlen, ds_varnames[i]))
        else:
            print("skipping variable {}/{}: {}".format(i+1, varlen, ds_varnames[i]))

    print("\n")
    return(ds_regrid)
