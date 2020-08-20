#!/bin/bash

# ___________________________________________________________________________________ #
function show_help {
    echo "usage:  $BASH_SOURCE -r SUPERVISED RATIO"
    echo "    -r SUPERVISED RATIO"
}

# default parameters
RATIO=1.0

while getopts ":m:r:" arg; do
  case $arg in
    r) RATIO=$OPTARG;;
    *) 
        echo "invalide option" 1>&2
        show_help
        exit 1
        ;;
  esac
done

# ___________________________________________________________________________________ #
# ___________________________________________________________________________________ #
scales="-a 1.357143 -b 1.214286 -g 1.000000"
phi=(
    "-p 1.2" \
    "-p 1.4" \
    "-p 1.6" \
    "-p 1.8" \
    "-p 2.0" \
    "-p 2.2" \
    "-p 2.4" \
    "-p 2.6" \
    "-p 2.8" \
    "-p 3.0" \
)

# ___________________________________________________________________________________ #
supervised_ratio="--supervised_ratio ${RATIO}"

run_number=0
for i in ${!phi[*]}
do
    run_number=$(( $run_number + 1 ))
   
    echo python copoundScaling_0.py -t 1 2 3 4 5 6 7 8 9 -v 10 ${scales} ${phi[$i]} ${supervised_ratio}
    python copoundScaling_0.py -t 1 2 3 4 5 6 7 8 9 -v 10 ${scales} ${phi[$i]} ${supervised_ratio}
done