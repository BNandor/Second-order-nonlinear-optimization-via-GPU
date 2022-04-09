buildAndRun:
	if [ -f ./gd ];then  rm gd ;fi
	/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  $(NVCCFLAGS) -o gd -arch=sm_60  -Xptxas -v
	./gd

test:
	python3 run.py

analyze:
	python3 analysis.py
	