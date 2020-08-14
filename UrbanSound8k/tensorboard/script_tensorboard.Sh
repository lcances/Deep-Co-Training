if [ "$1" = "run" ]; then
	start_port=6006
	for dir in `ls ubs8k`
	do
		tensorboard --port ${start_port} --logdir ubs8k/$dir --host legolas.irit.fr &
		start_port=$(($start_port + 1))
	done
fi

if [ "$1" = "kill" ]; then
	pkill tensorboard
fi

