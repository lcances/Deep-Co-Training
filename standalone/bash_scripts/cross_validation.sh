#!/bin/bash

cross_validation() {
    DATASET=$1
    CROSSVAL=$2

    # prepare cross validation parameters
    # ---- default, no crossvalidation
    if [ "$DATASET" = "ubs8k" ]; then
        mvar="-t 1 2 3 4 5 6 7 8 9 -v 10"
    elif [ "$DATASET" = "esc10" ] || [ "$DATASET" = "esc50" ]; then
        mvar="-t 1 2 3 4 -v 5"
    elif [ "$DATASET" = "speechcommand" ]; then
        mvar="-t 1 -v 2" # fake array to ensure exactly one run. Nut used by SpeechCommand anyway"
    fi

    # if crossvalidation is activated
    if [ $CROSSVAL -eq 1 ]; then
        if [ "$DATASET" = "ubs8k" ]; then
            mvar=$(python -c "import DCT.util.utils as u; u.create_bash_crossvalidation(10)")
            IFS=";" read -a folds <<< $mvar

        elif [ "$DATASET" = "esc10" ] || [ "$DATASET" = "esc50" ]; then
            mvar=$(python -c "import DCT.util.utils as u; u.create_bash_crossvalidation(5)")
            IFS=";" read -a folds <<< $mvar
        fi
    fi

    echo $mvar
}
