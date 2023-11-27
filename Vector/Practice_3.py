import numpy as np
import matplotlib.pyplot as plt
import time 
from scipy.stats import pearsonr
def Q1(x,y):
    #두 벡터를 입력으로 받아 두 개의 수를 출력하는 파이썬 함수.
    #두 개의 수는 피어슨 상관게수와 코사인 유사도.
    #np.corrcoef와 spartial.distance.cosine 사용하지 않고 구연하기
    # 변수들이 이미 평균중심화 되었다면 두출력이 동일하고 그렇지 않다면 결과가 달라야함

    num = np.dot(x,y) #numerator
    den = np.linalg.norm(x) * np.linalg.norm(y)#denominator
    cos = num / den
    
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    num = np.dot(xm,ym)
    den = np.linalg.norm(xm) * np.linalg.norm(ym)
    cor = num / den
    
    return cor, cos

def TestQ1():
    n = 50
    a = np.random.randn(n)
    b = np.random.randn(n)
    r,c = Q1(a,b)
    print(r,np.corrcoef(a,b)[0,1])
    
def Test2Q1():
    n = 50
    offset = 10
    a = np.random.randn(n)+offset
    b = np.random.randn(n)
    
    #mean - center
    a_no = a - np.mean(a)
    b_no = b - np.mean(b)
    print("Without mean-center (should diff)")
    print(Q1(a,b),"\n")

    print("With mean-center (should same)")
    print(Q1(a_no,b_no),"\n")
    
def Q2():
    #상관관계와 코사인 유사도 사이의 차이
    #0에서 3까지의 정수를 가진 변수와 이 변수에 특정 오프셋을 더한 두 번째 변수를 만들기
    #시스템적으로 오프셋을 -50에서 +50까지 변경하는 시뮬레이션 
    #즉, 시뮬레이션의 첫번째 반복에서는 두 번째 변수가 [-50, -49, -48, -47]
    #for 루프에서 두 변수 간의 상관관계와 코사인 유사도를 계산하고 결과 저장
    #평균 오프셋이 상관관계와 코사인 유사성에 어떻게 영향을 주는지 보여주는 선그래프.
    
    # create the variables
    a = np.arange(4,dtype=float)
    offsets = np.arange(-50,51)

    # initialize the results
    results = np.zeros((len(offsets),2))

    # run the simulation!
    for i in range(len(offsets)):
        results[i,:] = Q1(a,a+offsets[i])

    plt.figure(figsize=(8,4))
    h = plt.plot(offsets,results)
    h[0].set_color('k')
    h[0].set_marker('o')
    h[1].set_color([.4,.4,.7])
    h[1].set_marker('s')

    plt.xlabel('Mean offset')
    plt.ylabel('r or c')
    plt.legend(['Pearson','Cosine sim.'])
    plt.show()
    
    
def Q3():
    #피어슨 상관계수를 계산하는 여러 가지 파이썬 함수가 존재
    #이 중 하나는 pearsonr이고 Scipy라이브러이의 stats모듈에 존재 
    #이 파일의 소스 코드(힌트: //func-tioname)를  열어서 파이썬 구현이 이장에서 소개한 공식과 어떻게 일치하는지 확실히 이해해보기
    
    # inspect the source code
    pearsonr  # push F12 after cursor on word
    pass

# a bare-bones correlation function
def rho(x,y):
  xm = x-np.mean(x)
  ym = y-np.mean(y)
  n  = np.dot(xm,ym)
  d  = np.linalg.norm(xm) * np.linalg.norm(ym)
  
  return n/d
def Q4():
    #나의 상관관게함수가 numpy의 corrcoef 함수보다 빠른지 for 문을 통해서 확인하기
    
    # experiment parameters
    numIters  = 1000
    varLength =  500

    # clock my custom-written function
    tic = time.time()
    for i in range(numIters):
        x = np.random.randn(varLength,2)
        rho(x[:,0],x[:,1])
    t1 = time.time() - tic


    # now for numpy's corrcoef function
    tic = time.time()
    for i in range(numIters):
        x = np.random.randn(varLength,2)
        pearsonr(x[:,0],x[:,1])
    t2 = time.time() - tic


    # print the results!
    # Note: time() returns seconds, so I multiply by 1000 for ms
    print(f'My function took {t1*1000:.2f} ms')
    print(f'   pearsonr took {t2*1000:.2f} ms')
        
def KClustering():
    n = 100
    data = np.random.randn(n,2)  
    
    k = 10
    
    ridx = np.random.choice(range(len(data)),k,replace=False)
    print(ridx)
    centroids = data[ridx,:]
    print(centroids)
    dists = np.zeros((data.shape[0], k))
    print(dists.shape)
    for ci in range(k):
        dists[:,ci] = np.sum((data-centroids[ci,:])**2,axis = 1)
    
    groupidx = np.argmin(dists,axis = 1)
    
    for ki in range(k):
        centroids[ki, : ] = [np.mean(data[groupidx == ki ,0]),np.mean(data[groupidx == ki, 1])]
        
if __name__=="__main__":
    # TestQ1()
    # Test2Q1()
    # Q2()
    KClustering()
    pass