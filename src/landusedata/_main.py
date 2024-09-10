import argparse
import os

from landusedata.luh2 import main as luh2main
from landusedata.landusepft import main as lupftmain

def _shared_arguments(parser):
    parser.add_argument(
        'regrid_target_file',
        help='target surface data file with desired grid resolution',
    )
    parser.add_argument(
        'luh2_static_file',
        help='luh2 static data file',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='overwrite existing output file, if it exists',
    )
    parser.add_argument(
        '-t', '--test',
        action='store_true',
        help='test mode: only regrid one variable',
    )
    return parser

def main(argv=None):

    # Define top level parser
    parser = argparse.ArgumentParser(description="FATES landuse data tool")

    # Target regrid file - is there a nice way to share this between subparsers?
    # parser.add_argument('regrid_target_file', help='target surface data file with desired grid resolution')

    # Define subparser option for luh2 or landuse x pft data tool subcommands
    subparsers = parser.add_subparsers(required=True, title='subcommands',
                                       help='landuse data tool subcommand options')
    luh2_parser = subparsers.add_parser('luh2', prog='luh2',
                                        help='generate landuse harmonization timeseries data output')
    lupft_parser = subparsers.add_parser('lupft', prog='lupft',
                                         help='generate landuse x pft static data map output')

    # Set the default called function for the subparser command
    luh2_parser.set_defaults(func=luh2main)
    lupft_parser.set_defaults(func=lupftmain)

    # Shared arguments
    luh2_parser = _shared_arguments(luh2_parser)
    lupft_parser = _shared_arguments(lupft_parser)

    # LUH2 subparser arguments
    luh2_parser.add_argument('luh2_states_file',
                             help = "full path of luh2 raw states file")
    luh2_parser.add_argument('luh2_transitions_file',
                             help = "full path of luh2 raw transitions file")
    luh2_parser.add_argument('luh2_management_file',
                             help = "full path of luh2 raw management file")
    luh2_parser.add_argument("-w", "--regridder_weights",
                             default = 'regridder.nc',
                             help = "filename of regridder weights to save")
    luh2_parser.add_argument("-b","--begin",
                             type = int,
                             default = None,
                             help = "beginning of date range of interest")
    luh2_parser.add_argument("-e","--end",
                             type = int,
                             default = None,
                             help = "ending of date range to slice")
    luh2_parser.add_argument("-o","--output",
                             default = 'LUH2_timeseries.nc',
                             help = "output filename")

    # Landuse x pft subparser arguments
    lupft_parser.add_argument('clm_luhforest_file',
                              help = "CLM5_current_luhforest_deg025.nc")
    lupft_parser.add_argument('clm_luhpasture_file',
                              help = "CLM5_current_luhpasture_deg025.nc")
    lupft_parser.add_argument('clm_luhother_file',
                              help = "CLM5_current_luhother_deg025.nc")
    lupft_parser.add_argument('clm_surface_file',
                              help = "CLM5_current_surf_deg025.nc")
    lupft_parser.add_argument("-o","--output",
                              default = 'fates_landuse_pft_map.nc',
                              help = "output filename")

    # Parse the arguments
    args = parser.parse_args(argv)

    # Only overwrite existing output if --overwrite specified
    args.output = os.path.realpath(args.output)
    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(f"Output file exists; specify --overwrite to overwrite: {args.output}")

    # Create output directory, if needed. Otherwise, check write access.
    output_directory = os.path.dirname(args.output)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    elif not os.access(output_directory, os.W_OK):
        raise PermissionError("No write permissions in " + output_directory)

    # Call the default function for the given subcommand
    args.func(args)

    # Return successful completion
    return 0

# Guard against import time side effects
if __name__ == '__main__':
    raise SystemExit(main())
