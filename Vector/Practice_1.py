import numpy as np
import matplotlib.pyplot as plt


def Q1():
    sp = -6
    ep = 6
    
    v = np.array([[0,0],[1,2]])
    w = np.array([[1,2],[4,-6]])
    w2 = np.array([[0,0],[4,-6]])
    
    plt.figure(1, figsize=(6,6))
    
    plt.axis('square')
    plt.axis([-3,3,-3,3])
    plt.xlabel('$v_0$')
    plt.ylabel('$v_1$')
    
    plt.xlim(sp, ep)          
    plt.ylim(sp, ep)
    plt.xticks(ticks=np.arange(sp, ep, step=1))
    plt.yticks(ticks=np.arange(sp, ep, step=1))
    plt.axhline(0, color='gray', alpha = 0.3)
    plt.axvline(0, color='gray', alpha = 0.3)
    plt.grid()

    plt.title("v+w")
    plt.arrow(v[0][0], v[0][1], v[1][0], v[1][1], head_width = .2, head_length = .2,color='r',length_includes_head=True)
    plt.arrow(w[0][0], w[0][1], w[1][0], w[1][1], head_width = .2, head_length = .2,color='b',length_includes_head=True)
    plt.arrow(v[0][0], v[0][1], v[1][0]+w[1][0], v[1][1]+w[1][1], head_width = .2, head_length = .2,color='g',length_includes_head=True)
    
    plt.figure(2,figsize=(6,6))
    plt.xlim(sp, ep)          
    plt.ylim(sp, ep)
    plt.xticks(ticks=np.arange(sp, ep, step=1))
    plt.yticks(ticks=np.arange(sp, ep, step=1))
    plt.axhline(0, color='gray', alpha = 0.3)
    plt.axvline(0, color='gray', alpha = 0.3)
    plt.grid()

    plt.title('v-w')

    plt.arrow(v[0][0], v[0][1], v[1][0], v[1][1], head_width = .2, head_length = .2,color='r',length_includes_head=True)
    plt.arrow(w2[0][0], w2[0][1], w2[1][0], w2[1][1], head_width = .2, head_length = .2,color='b',length_includes_head=True)
    plt.arrow(w2[1][0], w2[1][1], v[1][0]-w2[1][0], v[1][1]-w2[1][1], head_width = .2, head_length = .2,color='g',length_includes_head=True)
    #plt.savefig('Figure_01_01.png',dpi=300)
    plt.show()
def Q2():
    v = np.array([6,4,5])
    v_norm = sum(v**2)**(0.5)
    print(f'v_norm = {v_norm} | np.linalg.norm = {np.linalg.norm(v)}')
    print(v_norm==np.linalg.norm(v))
def Q3():
    v = np.array([int(input(f'input {i+1} column number = ')) for i in range(2) ])
    
    if((sum(v*v)**0.5)==0):
        print("v norm is 0 . so Can't divide")
        return
    unit_v = v/(sum(v*v)**0.5)
    
    plt.figure(figsize=(20,20))
    plt.xlim(-10, 10)          
    plt.ylim(-10, 10)
    plt.xticks(ticks=np.arange(-10, 10, step=1))
    plt.yticks(ticks=np.arange(-10, 10, step=1))

    plt.arrow(0, 0, v[0], v[1], head_width = .2, head_length = .2,color='r',length_includes_head=True)
    plt.arrow(0, 0,unit_v[0], unit_v[1], head_width = .2, head_length = .2,color='b',length_includes_head=True)
    #for i in args:
    #    plt.arrow(0, 0, i[0], i[1], head_width = .2, head_length = .2,color=any)
    #plt.arrow(0, 0, b[0], b[1], head_width = .5, head_length = .5, color = 'blue')
    #plt.plot(x, y)
    #plt.annotate('Max Value', xy=(3, 9), xytext=(2.5, 12), arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.axhline(0, color='gray', alpha = 0.3)
    plt.axvline(0, color='gray', alpha = 0.3)
    plt.title("Vector")
    plt.grid()
    plt.show()
def Q4() ->np.ndarray:
    v = np.array([int(input(f"{i+1}. input vector number :")) for i in range(2)])
    s = int(input("input size : "))
    v_norm = v/(sum(v*v)**0.5)
    return v_norm*s
def Q5():
    print("rowv2colv")
    n = int(input("input vector size : "))
    v = np.array([int(input(f"{i+1}. input vector number :")) for i in range(n)])
    v_t =np.empty((n, 1))
    for idx,j in enumerate(v):
        v_t[idx][0]=j
    print(v_t)
def Q6():
    v = np.array([int(input(f"{i+1}. input vector number :")) for i in range(2)])
    v_norm = sum(v*v)**0.5
    vv_norm = v_norm**2
    v_dot = np.dot(v,v)
    print(f'vv_norm = {vv_norm} |v_dot = {v_dot} | is same? {vv_norm == v_dot}')
def Q7():
    a,b = [np.array([int(input(f"{i+1}-{j+1}. input vector number :")) for j in range(2)]) for i in range(2)]
    print(f'np.dot(a,b)==np.dot(b,a) =>{np.dot(a,b)==np.dot(b,a)}')
def Q8():
    a,b = [np.array([float(input(f"{i+1}-{j+1}. input vector number :")) for j in range(2)]) for i in range(2)]
    plt.figure(figsize=(10,10))
    plt.xlim(-5, 5)          
    plt.ylim(-5, 5)
    plt.xticks(ticks=np.arange(-5, 5, step=1))
    plt.yticks(ticks=np.arange(-5, 5, step=1))
    
    
    beta = np.dot(a,b)/np.dot(a,a)
    a_beta = beta*a
    plt.arrow(0, 0, a[0], a[1], head_width = .2, head_length = .2,color='b',length_includes_head=True)
    plt.arrow(0, 0, b[0], b[1], head_width = .2, head_length = .2,color='r',length_includes_head=True)
    #plt.arrow(a_beta[0], a_beta[1], b[0]-a_beta[0], b[1]-a_beta[1], head_width = .2, head_length = .2,color='g',length_includes_head=True)
    # plt.plot(a*beta, b, linestyle=(0, (5, 1)), color='C0', label='(0, (5, 1))')
    #plt.annotate('Max Value', xy=(3, 9), xytext=(2.5, 12), arrowprops=dict(facecolor='black', arrowstyle='->'))

    # projection vector
    plt.plot([b[0],beta*a[0]],[b[1],beta*a[1]],'k--')

    # projection on a
    plt.plot(beta*a[0],beta*a[1],'ko',markerfacecolor='w',markersize=13)
    
    # add labels
    plt.text(a[0]+.1,a[1],'a',fontweight='bold',fontsize=18)
    plt.text(b[0],b[1]-.3,'b',fontweight='bold',fontsize=18)
    plt.text(beta*a[0]-.35,beta*a[1],r'beta',fontweight='bold',fontsize=18)
    plt.text((b[0]+beta*a[0])/2,(b[1]+beta*a[1])/2+.1,r'(b-beta_a)',fontweight='bold',fontsize=18)

    plt.axhline(0, color='gray', alpha = 0.3)
    plt.axvline(0, color='gray', alpha = 0.3)
    plt.title("Vector")
    plt.grid()
    plt.show()
def Q9():
    r,t = [np.array([float(input(f"{i+1}-{j+1}. input vector number :")) for j in range(2)]) for i in range(2)]
    
    beta = np.dot(r,t)/np.dot(r,r)
    r_beta = beta*r
    orthogonal_t= [t[i]-r_beta[i] for i in range(2)]
    
    print(np.dot(orthogonal_t,r_beta))
    plt.figure(figsize=(10,10))
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.xticks(ticks=np.arange(-5,5,step=1))
    plt.yticks(ticks=np.arange(-5,5,step=1))
    
    plt.arrow(0, 0, r[0], r[1], head_width = .2, head_length = .2,color='b',length_includes_head=True)
    plt.arrow(0, 0, t[0], t[1], head_width = .2, head_length = .2,color='r',length_includes_head=True)
    
    plt.plot([0,orthogonal_t[0]],[0,orthogonal_t[1]],'k--')
    plt.plot([0,r_beta[0]],[0,r_beta[1]],'k--')
    
    plt.axhline(0, color='gray', alpha = 0.3)
    plt.axvline(0, color='gray', alpha = 0.3)
    plt.title("Vector")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    #Q1()
    #Q2()
    #Q3()
    #print(Q4())
    #Q5()
    #Q6()
    #Q7()
    #Q8()
    Q9()
