
if [ ! $# -eq 1 ]
then
	echo "usage: $@ problem"
	exit 1
fi	
logPath=SNLP/problems/$1/logs/$1.log
csvPath=SNLP/problems/$1/csv/$1.csv
cd ..
./buildRun.sh > $logPath
cat  $logPath | grep xCurrent | sed 's/xCurrent//g' > $csvPath
cd -

