
![alt text](img/Logo.png)
<br>

##  NMHH: Nested Markov Chain Hyper Heuristic:
A hyperheuristic framework for the continuous domain.

###  Requirements:


Install required packages via

```
pip3 install -r requirements.txt 
```
- Python >= 3.8 is recommended

- A CUDA enabled GPU supporting at least ```sm_arch=60``` and the CUDA compiler (`nvcc >= 11.4`) from the [cuda toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) is required.


### Optimize benchmark functions with:

```shell
./startexp.sh
```
- the experiment results can be found under ```hhanalysis/logs```
- the results are cached, these need to be deleted for the experiments to be performed again