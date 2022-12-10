buildAndRun:
	if [ -f ./gd ];then  rm gd ;fi
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  $(NVCCFLAGS) -o gd -arch=sm_60  -Xptxas -v
	./gd

test:
	python3 run.py

#http://www.sfu.ca/~ssurjano/rosen.html min 0
test-rosenbrock:
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  -DOPTIMIZER=GD           -DSAFE      -DPROBLEM_ROSENBROCK                          -DFRAMESIZE=400                   -DGLOBAL_SHARED_MEM -DITERATION_COUNT=10000 -DPOPULATION_SIZE=100 -DPRINT -DX_DIM=1000 -o gd -arch=sm_60  -Xptxas -v  && ./gd

#http://www.sfu.ca/~ssurjano/stybtang.html min 39.165d
test-styblinskitang: 
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  -DOPTIMIZER=GD            -DSAFE     -DPROBLEM_STYBLINSKITANG                          -DFRAMESIZE=400                   -DGLOBAL_SHARED_MEM -DITERATION_COUNT=1000 -DPOPULATION_SIZE=100 -DPRINT -DX_DIM=1000 -o gd -arch=sm_60  -Xptxas -v  && ./gd

#http://www.sfu.ca/~ssurjano/trid.html min -d(d+4)(d-1)/6
test-trid:
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  -DOPTIMIZER=GD               -DSAFE  -DPROBLEM_TRID                         -DFRAMESIZE=400                   -DGLOBAL_SHARED_MEM -DITERATION_COUNT=1000 -DPOPULATION_SIZE=50 -DPRINT -DX_DIM=100 -o gd -arch=sm_60  -Xptxas -v  && ./gd

#http://www.sfu.ca/~ssurjano/rastr.html min 0
test-rastrigin:
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  -DOPTIMIZER=GD              -DSAFE     -DPROBLEM_RASTRIGIN                         -DFRAMESIZE=400                   -DGLOBAL_SHARED_MEM -DITERATION_COUNT=1000 -DPOPULATION_SIZE=30 -DPRINT -DX_DIM=100 -o gd -arch=sm_60  -Xptxas -v  && ./gd

# min 0
test-schwefel223: 
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  -DOPTIMIZER=GD              -DSAFE   -DPROBLEM_SCHWEFEL223                         -DFRAMESIZE=400                   -DGLOBAL_SHARED_MEM -DITERATION_COUNT=1000 -DPOPULATION_SIZE=30 -DPRINT -DX_DIM=1000 -o gd -arch=sm_60  -Xptxas -v  && ./gd

analyze:
	python3 analysis.py