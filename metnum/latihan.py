i = 0
x = 0.5
app = []
tv = 1.648721
def faktorial(n):
    a = 1
    for i in range(1, n+1):
        a*=i
    return a

while True:
    if i == 0:
        app.append(1)
        print('terms =', i+1)
        print('result =', app[-1])
        print('et =', ((tv - app[-1])/tv)*100)
    else:
        app.append(app[-1]+((float(x)**i)/faktorial(i)))
        print('terms =', i+1)
        print('result =',app[-1])
        print('et =', ((tv - app[-1])/tv)*100)
        print('ea =', ((app[-1]-app[-2])/app[-1])*100)
        if ((app[-1]-app[-2])/app[-1])*100 <0.05:
            break
    i+=1
    print('====================================================\n\n')




