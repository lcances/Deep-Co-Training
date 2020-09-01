parse_long() {
    if [ "$1" ]; then
        echo $1
    else
        echo "missing argument" >&2
        exit 1
    fi
}

while :; do
    if ! [ "$1" ]; then break; fi

    echo "1 is : " $1

    case $1 in
        -h | --help)
            echo "this is the help !!!!!"
            exit 1
            ;;
        -m | --model)
            echo "this is the model argument" $2
            MODEL=$(parse_long $2)
            shift; shift
            ;;
        -d | --dataset)
            echo "this the dataset argument" $2
            DATASET=$(parse_long $2)
            shift; shift
            ;;
    esac
done

echo "MODEL = ${MODEL}"
echo "DATASET = ${DATASET}"
