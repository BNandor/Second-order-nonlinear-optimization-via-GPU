rm gd
/usr/local/cuda-11.4/bin/nvcc main.cu  -g -G  -o gd -arch=sm_60  -Xptxas -v
./gd

