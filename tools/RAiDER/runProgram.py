import copy

from RAiDER.delay import tropo_delay
from RAiDER.cli.raiderDelay import parseCMD, read_template_file
from RAiDER.checkArgs import checkArgs


# this and the next function are modified from MintPy
def _tropo_delay(args):

    args_copy = copy.deepcopy(args)

    if 0 < len(args['times']) < 2:
        args_copy['times'] = args['times'][0]
        try:
            (_, _) = tropo_delay(args_copy)
        except RuntimeError:
            logger.exception("Date %s failed", args_copy['times'])
    else:
        for tim, wetFilename, hydroFilename in zip(args['times'], args['wetFilenames'], args['hydroFilenames']):
            try:
                args_copy['times'] = tim
                args_copy['wetFilenames'] = wetFilename
                args_copy['hydroFilenames'] = hydroFilename
                (_, _) = tropo_delay(args_copy)
            except RuntimeError:
                logger.exception("Date %s failed", tim)
                continue


##########################################################################
def main(iargs=None):
    # parse
    inps = parseCMD(iargs)
    
    # Read the template file
    args = read_template_file(inps.customTemplateFile)
    
    # Argument checking
    args = checkArgs(args)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # run
    _tropo_delay(args)


###########################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
