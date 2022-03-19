import random

n=100
maxDist = 100
for i in range(1,n+1):
    a=i-1
    # b=random.randint(0,n-1)
    b=i
    while a == b:
        b=random.randint(0,n-1)
    d=random.random()*maxDist
    print(f"{a} {b} {d}")
