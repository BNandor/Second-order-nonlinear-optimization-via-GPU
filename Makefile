buildAndRun:
	if [ -f ./gd ];then  rm gd ;fi
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  $(NVCCFLAGS) -o gd -arch=sm_60  -Xptxas -v
	./gd

test:
	python3 run.py

test-rosenbrock:
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  -DOPTIMIZER=GD                 -DPROBLEM_ROSENBROCK                          -DFRAMESIZE=400                   -DGLOBAL_SHARED_MEM -DITERATION_COUNT=10000 -DPOPULATION_SIZE=30 -DPRINT -DX_DIM=100 -o gd -arch=sm_60  -Xptxas -v  && ./gd
analyze:
	python3 analysis.py
	