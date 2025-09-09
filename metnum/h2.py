import  math as m

def bisection(a, b):
    return (a+b)/2

def fungsi(c):
    fx = ((68.1*9.8)/c)*(1-(m.e**(-10*c/68.1)))-40
    return fx

awal = -100
akhir = 100
hs = fungsi(awal)
hf = fungsi(akhir)

for i in range(100):
    bs = bisection(awal, akhir)
    hb = fungsi(bs)
    if (hb <= abs(hs)):
        awal = bs
