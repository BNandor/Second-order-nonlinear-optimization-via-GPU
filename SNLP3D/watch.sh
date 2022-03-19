
if [ ! $# -eq 1 ]
then
	echo "usage: $@ problem"
	exit 1
fi

cd ..
logPath=SNLP3D/problems/$1/logs
if [ ! -d $logPath ];
then
	mkdir -p $logPath
fi

logFile=$logPath/$1.log
csvPath=SNLP3D/problems/$1/csv
if [ ! -d $csvPath ];
then
	mkdir -p $csvPath
fi
csvFile=$csvPath/$1.csv
fcsvFile=$csvPath/$1.f.csv
./buildRun.sh > $logFile
cat  $logFile | grep xCurrent | sed 's/xCurrent//g' > $csvFile
cat $logFile | grep "^f:" |sed "s/f://g" > $fcsvFile
cd -

