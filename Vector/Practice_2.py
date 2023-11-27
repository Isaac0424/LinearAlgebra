import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random

def Q1():
    # ## the scalars
    # l1 = 1
    # l2 = 2
    # l3 = -3

    # # the vectors
    # v1 = np.array([4,5,1])
    # v2 = np.array([-4,0,-4])
    # v3 = np.array([1,3,2])

    # # linear weighted combination
    # l1*v1 + l2*v2 + l3*v3
    n = 3 
    scalars = [float(input(f"input {i}.scalar : ")) for i in range(n)]
    vectors=[np.array([float( input(f"{j}.input vector[{i}] : ")) for i in range(n)]) for j in range(n)]

    linear_combination = np.zeros(n)
    
    for s,v in zip(scalars,vectors):
        linear_combination +=s*v    
        
    print(f'np linear combination ={linear_combination} ')

def Q2():
    n = 3 
    scalars = [float(input(f"input {i}.scalar : ")) for i in range(n+1)]
    vectors=[np.array([float( input(f"{j}.input vector[{i}] : ")) for i in range(n)]) for j in range(n)]

    linear_combination = np.zeros(n)
    
    for i in range(len(scalars)):
        linear_combination += scalars[i]*vectors[i]    
        
    print(f'np linear combination ={linear_combination} ')
    # invalid index ERROR
    
def Q3():
    A  = np.array([ 1,3 ])
    xlim = [-4,4]
    
    scalars = np.random.uniform(low=xlim[0],high=xlim[1],size=100)
    plt.figure(figsize=(6,6))
    for s in scalars:
        p = A*s
        plt.plot(p[0],p[1],'ko')
   
    plt.xlim(xlim)
    plt.ylim(xlim)
    plt.grid()
    plt.text(-4.5,4.5,'A)',fontweight='bold',fontsize=18)
    plt.savefig('Figure_02_07a.png',dpi=300)
    plt.show()

def Q4():
    xlim = [-4,4]
    v1 = np.array([3,5,1])
    v2 = np.array([0,2,2])

    scalars = np.random.uniform(low=xlim[0],high=xlim[1],size=(100,2))
    
    points = np.zeros((100,3))
    
    for i in range(len(scalars)):
        points[i] = scalars[i][0]*v1+scalars[i][1]*v2
        
    fig = go.Figure( data=[go.Scatter3d(
                            x=points[:,0],y=points[:,1],z=points[:,2],
                            mode='markers')])
    fig.show()

def Q5():
    pass     
if __name__ == '__main__':
    #Q1()
    Q4()