import random

n=1000
maxDist = 100
for i in range(n):
    a=i
    b=random.randint(0,n-1)
    while a == b:
        b=random.randint(0,n-1)
    d=random.random()*maxDist
    print(f"{a} {b} {d}")
