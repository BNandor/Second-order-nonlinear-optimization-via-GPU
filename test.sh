
./buildRun.sh
rm $2
if [  ! $# -eq 2 ];
then
	echo "usage $0 testCount measurementsPath"
	exit 1	
fi

for i in `seq $1`;
do
	echo "Test: $i"
	./gd | grep "time" | sed "s/\(.*\): \(.*\)/\2/g">> $2	
done

