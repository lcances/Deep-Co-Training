#!/bin/bash

append() {
#     echo $#
    # if the to check is empty and not surrounded by parenthesis, 
    # It will not be counted as parameters. If it is surrounded 
    # by parenthesis, then it will be. therefore the double verification
    
    if [ "$#" -eq 3 ]
    then
        msg=$1
        to_add=$2
        key=$3

        if [ -n "${to_add}" ]
        then
            echo "$msg $key $to_add"
            
        else
            echo "$msg"
        fi
    
    else
        echo "$1"
    fi
}

# # example
# orig="origin"
# a="test"
# b="test2"

# one=$(append "$orig" $a --a)
# one=$(append "$one" $b --b)

# echo $one