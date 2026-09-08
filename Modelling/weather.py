import numpy as np
from matplotlib import pyplot as plt
A=np.array([[0.5, 0.5, 0.25], [0.25, 0.0, 0.25], [0.25, 0.5, 0.5]])
print("A:")
print(A)
Xtoday=np.array([[1], [0], [0]])
print("Xtoday:")
print(Xtoday)
print(A@Xtoday)
the_weather=np.zeros((3, 50))
the_weather[:,0]=Xtoday.flatten()
for k in range(50):
    Xtomorrow=A@Xtoday
    the_weather[:,k]=Xtomorrow.flatten()
    print(f"Xtomorrow:", Xtomorrow)
    Xtoday=Xtomorrow
plt.plot(the_weather.transpose())
plt.grid(True)
plt.legend()
plt.show()