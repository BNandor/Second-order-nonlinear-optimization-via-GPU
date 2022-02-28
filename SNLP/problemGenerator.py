import random

n=100
maxDist = 10
for i in range(n):
    a=i
    b=random.randint(0,n-1)
    while a == b:
        b=random.randint(0,n-1)
    d=random.random()*maxDist
    print(f"{a} {b} {d}")
