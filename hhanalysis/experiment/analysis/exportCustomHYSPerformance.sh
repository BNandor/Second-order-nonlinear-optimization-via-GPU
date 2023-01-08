resultsDir=`echo ${PWD}/results/CustomHYS`

customHYSResults=$resultsDir/results.json
perfDir=../../logs/CustomHYSPerf
cd $perfDir

echo "{" > $customHYSResults

for ldir in `ls`
do
        if [  -d $ldir ]
        then
                cd $ldir
                echo \"$ldir\": '[' >> $customHYSResults
                for log in `ls | sort -n`
                do
                        echo '{' '"step":"'$log'","med_iqr":' `cat $log | sed 's/, \"details.*//g' | sed 's/.*\"performance\"://g'` '},'>> $customHYSResults
                done
                echo '],' >> $customHYSResults
                cd ..
        fi
done
echo '}' >> $customHYSResults