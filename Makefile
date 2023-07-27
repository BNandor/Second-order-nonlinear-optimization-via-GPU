buildAndRun:
	if [ -f ./opt ];then  rm opt ;fi
	export LD_LIBRARY_PATH=$(ROOT)/libs && export CPATH=$(ROOT)/libs/include/nomad:$(ROOT)/libs/include/eigen3:$(ROOT)/libs/include/cmaes  && /usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  $(NVCCFLAGS) -L$(ROOT)/libs  -lcmaes -lsgtelib -lnomadAlgos -lnomadEval -lnomadUtils -o opt -arch=sm_60  
	echo "Staring optimizer" &&  ./opt

test:
	python3 run.py

#http://www.sfu.ca/~ssurjano/rosen.html min 0
test-rosenbrock:
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  -DSAFE -DPROBLEM_ROSENBROCK -DHYPER_LEVEL_TRIAL_SAMPLE_SIZE=30 -DITERATION_COUNT=1000 -DPOPULATION_SIZE=30 -DX_DIM=3 -DHH_TRIALS=10 -DLOGS_PATH='"hhanalysis/logs/rosenbrock.json"' -o gd -arch=sm_60  -Xptxas -v  && ./gd

#http://www.sfu.ca/~ssurjano/stybtang.html min -39.165d
test-styblinskitang: 
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  -DSAFE -DPROBLEM_STYBLINSKITANG     -DHYPER_LEVEL_TRIAL_SAMPLE_SIZE=30 -DITERATION_COUNT=1000 -DPOPULATION_SIZE=30  -DX_DIM=100 -DHH_TRIALS=10 -DLOGS_PATH='"hhanalysis/logs/styblinskitang.json"' -o gd -arch=sm_60  -Xptxas -v  && ./gd

#http://www.sfu.ca/~ssurjano/trid.html min -d(d+4)(d-1)/6
test-trid:
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  -DSAFE -DPROBLEM_TRID    -DHYPER_LEVEL_TRIAL_SAMPLE_SIZE=30 -DITERATION_COUNT=1000 -DPOPULATION_SIZE=50  -DX_DIM=100 -DHH_TRIALS=10 -DLOGS_PATH='"hhanalysis/logs/trid.json"' -o gd -arch=sm_60  -Xptxas -v  && ./gd

#http://www.sfu.ca/~ssurjano/rastr.html min 0
test-rastrigin:
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  -DSAFE -DPROBLEM_RASTRIGIN    -DHYPER_LEVEL_TRIAL_SAMPLE_SIZE=30 -DITERATION_COUNT=1000 -DPOPULATION_SIZE=30  -DX_DIM=100 -DHH_TRIALS=10 -DLOGS_PATH='"hhanalysis/logs/rastrigin.json"' -o gd -arch=sm_60  -Xptxas -v  && ./gd

# min 0
test-schwefel223: 
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  -DSAFE -DPROBLEM_SCHWEFEL223    -DHYPER_LEVEL_TRIAL_SAMPLE_SIZE=30 -DITERATION_COUNT=1000 -DPOPULATION_SIZE=30  -DX_DIM=1000 -DHH_TRIALS=10 -DLOGS_PATH='"hhanalysis/logs/schwefel223.json"' -o gd -arch=sm_60  -Xptxas -v  && ./gd

# min 0 at x(i)=+-sqrt(i) i in 1,n
test-qing: 
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  -DSAFE -DPROBLEM_QING    -DHYPER_LEVEL_TRIAL_SAMPLE_SIZE=30 -DITERATION_COUNT=1000 -DPOPULATION_SIZE=30  -DX_DIM=100 -DHH_TRIALS=10 -DLOGS_PATH='"hhanalysis/logs/qing.json"' -o gd -arch=sm_60  -Xptxas -v  && ./gd

analyze:
	python3 analysis.py