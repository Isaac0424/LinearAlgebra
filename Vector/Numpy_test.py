import numpy as np

def PrintVector():
    print("PrintVector()")
    asList = [1,2,3]
    asArray = np.array ([1,2,3])
    rowVec = np.array([[1,2,3]])
    colVec = np.array([[1],[2],[3]])
    print(asList)
    print( asArray)
    print(rowVec)
    print(colVec)
    print(f'asList : {np.shape(asList)}')
    print(f'asArray : {asArray.shape}')
    print(f'rowVec : {rowVec.shape}')
    print(f'colVec : {colVec.shape}')

def SumVector():
    print("SumVector()")
    v = np.array([4,5,6])
    w = np.array([10,20,30])
    u = np.array([0,3,6,9])
    
    vPlusW = v + w 
    # uPlusW = u + w  차원 불일치
    
    print(f'v({v}) + w({w}) =  vPlusW({vPlusW})')
    
def BroadSumVec():
    print("BroadSumVec()")
    v = np.array([[4,5,6]])
    w = np.array([[10,20,30]]).T
    print(v+w)
    
def MulVec():
    print("MulVec()")
    s = 2
    a = [3,4,5]
    b = np.array(a)
    
    print(f's({s}) * list_a({a}) = s*a({s*a})')
    print(f's({s}) * vector_b({b}) = s*b({s*b})')

def Norm():
    print("Norm()")
    v = np.array([1,2,3,4,5,6,7,8,9])
    v_dim = len(v)
    v_mag = np.linalg.norm(v)
    print(f'v ={v}')
    print(f'v dimension = {v_dim}')
    print(f'v Norm = {v_mag}')
    
def DotProduct():
    print("DotProduct()")
    v = np.array([1,2,3,4])
    w = np.array([5,6,7,8])
    d = np.dot(v,w)

    print(f'v({v}) | w({w})')
    print(f'np.dot(v,w)= ({d})')
    s = 10
    print(f'np.dot(s({s})*v,w) = {np.dot(s*v,w)}')

    a=np.array([0,1,2])
    b=np.array([3,5,8])
    c=np.array([13,21,34])
    
    res1 = np.dot( a, b+c)
    res2 = np.dot(a,b) + np.dot(a,c)
    print(f'a = {a} , b = {b}, c = {c}')
    print(f'np.dot (a,b+c) = {res1}')
    print(f'np.dot(a,b) + np.dot(a,c) = {res2}')

def HardamardProduct():
    print("Hardamard Product")
    a = np.array([5,4,8,2])
    b = np.array([1,9,.5,-1])
    print(f'a({a})*b({b}) = {a*b}')


if __name__ == "__main__":
    SumVector()
    BroadSumVec()
    MulVec()
    Norm()
    DotProduct()
    HardamardProduct()