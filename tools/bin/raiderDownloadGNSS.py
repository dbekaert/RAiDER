#!/u/leffe0/ssangha/tools/conda_installation/stable_feb9_2020/envs/RAiDER/bin/python
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Simran Sangha, Jeremy Maurer, & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from RAiDER.downloadGNSSDelays import cmdLineParse,queryRepos

if __name__ == "__main__":
    inps = cmdLineParse()

    queryRepos(inps)
