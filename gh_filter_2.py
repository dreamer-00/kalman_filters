from turtle import color
import numpy
import matplotlib.pyplot as plt
def g_h_filter(data, x0, dx, g, h, dt=1, pred=None):
    '''data contains data to be filtered
       x0 is the initial value of our state variable
       dx is the initial change rate of our state variable
       dt is the rate of change of time step
       pred is set to None if the list is not given'''
    x=x0
    result=[]
    for z in data :
        x_est=x+(dx*dt)
        dx=dx
        if pred is not None:
            pred.append(x_est)
        residual=z-x_est
        dx=dx + h*(residual)/dt
        x=x_est+g*residual
        result.append(x)
    return result
weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]

filtered_data=g_h_filter(data=weights, x0=160.0, dx=1.0, g=0.4, h=0.33, dt=1.0)
plt.figure(figsize=(10, 5))
plt.plot(filtered_data, '--', color='red', label='Filtered Estimate(x)')
plt.plot(weights, '-', color='blue', label='Measurement(z)')
plt.plot([0, len(weights)-1], [160, 160+len(weights)-1], color='green', label='True Actual trend')
plt.title('Generic g-h filter performance')
plt.xlabel('Time Step (k)')
plt.ylabel('State value (x)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='best')
plt.tight_layout()
plt.show()
