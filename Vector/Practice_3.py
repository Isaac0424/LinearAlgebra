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
    n = 150
    k = 3
    data = np.random.randn(n,2)  #n *2의 가우시안 표준 정규분포 난수 생성 분포 평균은 0이고 분산은 1이다.
    
    ridx = np.random.choice(range(len(data)),k,replace=False) # 0~len(data) 만큼 k개 , 중복 없이!! 랜덤으로 고르기 
    print(ridx)
    centroids = data[ridx,:] # data행렬은 특징별 샘플링 
    print(centroids)
    
    a=2
    b=2
    # setup the figure
    fig,axs = plt.subplots(a,b,figsize=(6,6))
    axs = axs.flatten()
    lineColors = [ [0,0,0],[.4,.4,.4],[.8,.8,.8] ]#'rbm'

    # plot data with initial random cluster centroids
    axs[0].plot(data[:,0],data[:,1],'ko',markerfacecolor='w')
    axs[0].plot(centroids[:,0],centroids[:,1],'ko')
    axs[0].set_title('Iteration 0')
    axs[0].set_xticks([])
    axs[0].set_yticks([])


    # loop over iterations
    for iteri in range(a*b-1):
        dists = np.zeros((data.shape[0], k)) #n * k
        print(dists.shape)
        for ci in range(k):
            dists[:,ci] = np.sum((data-centroids[ci,:])**2,axis = 1)
        #print(dists)
        groupidx = np.argmin(dists,axis = 1)
        print(groupidx)
        
        for ki in range(k):
            centroids[ki, : ] = [np.mean(data[groupidx == ki ,0]),np.mean(data[groupidx == ki, 1])]
        # plot data points
        for i in range(len(data)):
            axs[iteri+1].plot([ data[i,0],centroids[groupidx[i],0] ],[ data[i,1],centroids[groupidx[i],1] ],color=lineColors[groupidx[i]])
        axs[iteri+1].plot(centroids[:,0],centroids[:,1],'ko')
        axs[iteri+1].set_title(f'Iteration {iteri+1}')
        axs[iteri+1].set_xticks([])
        axs[iteri+1].set_yticks([])
    plt.show()
def Q5():
    kernel = np.array([-1,1])

    # and the "signal" 
    signal = np.zeros(30)
    signal[10:20] = 1

    result = np.zeros(len(signal))
    
    for i in range(len(signal)-(len(kernel)-1)):
        result[i] = np.dot(kernel,signal[i:i+len(kernel)])   

    # plot them
    _,axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].plot(kernel,'ks-')
    axs[0].set_title('Kernel')
    axs[0].set_xlim([-15,15])
    
    axs[1].plot(signal,'ks-')
    axs[1].set_title('Time series signal')
    
    axs[2].plot(signal,'s-',color=[0,0,0])
    axs[2].plot(result,'ks-',color=[.5,.5,1])
    axs[2].set_title('filtering')
    plt.show()
    
    print(result)
    
def Q6():
    # define the kernel (a sorta-kinda Gaussian)
    kernel = np.array([0,.1,.3,.8,1,.8,.3,.1,0])
    kernel = kernel / np.sum(kernel)

    # some handy length parameters
    Nkernel = len(kernel)
    halfKrn = Nkernel//2


    # and the signal
    Nsignal = 100
    timeseries = np.random.randn(Nsignal)

     # make a copy of the signal for filtering
    filtsig = timeseries.copy()

    # loop over the signal time points
    for t in range(halfKrn+1,Nsignal-halfKrn):
        filtsig[t] = np.dot(kernel,timeseries[t-halfKrn-1:t+halfKrn])


    # plot them
    _,axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].plot(kernel,'ks-')
    axs[0].set_title('Kernel')
    axs[0].set_xlim([-1,Nsignal])

    axs[1].plot(timeseries,'ks-')
    axs[1].set_title('Time series signal')

    axs[2].plot(timeseries,color='k',label='Original',linewidth=1)
    axs[2].plot(filtsig,'--',color=[.6,.6,.6],label='Smoothed',linewidth=2)
    axs[2].legend()

    plt.show()


if __name__=="__main__":
    # TestQ1()
    # Test2Q1()
    # Q2()
    KClustering()
    #Q5()
    #Q6()
    pass