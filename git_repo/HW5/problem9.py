import numpy as np
import matplotlib.pyplot as plt

#constants

def HMC(U,K,dU,N,x_0, p_0, epsilon=0.01, L=100):
    path = [x_0]
    print("path",path)
    current_x = x_0
    current_p = p_0
    
    C = np.array([[250.25, -249.75],[-249.75, 250.25]]) # corr^-1
    n = C.shape[0]  

    def evalE(_x):

        return np.dot(_x,np.dot(C,_x)) / 2.
    
    def evalG(_x):
        return np.dot(C,_x)

    H = np.zeros(N)
    xall = np.zeros((N,2))
    accept=0
    for j in range(N):
        x = current_x
        p = current_p
        
        G_new = evalG(current_x)
        #draw a new p
        p = np.random.randn(n)
        H = np.dot(p,p) / 2. + evalE(current_x)
        #print(p)
        current_p=p
        
        # leap frog
        
        # Make a half step for momentum at the beginning
        #p = p - epsilon*dU(x)/2.0
        p = p - [t * epsilon for t in dU(x)]
        #print(p)
        # alternate full steps for position and momentum
        tmp = []
        for i in range(L):
            #print(x)
            p = p - epsilon*G_new/2.
            x = x+epsilon*p
            tmp.append([x[0],x[1]])
            G_new = evalG(x)
            if (i != L-1):
                #VER 1: p = p - np.array([a * epsilon for a in dU(x)])
                #VER2:
                p = p - epsilon*G_new/2. 
        #make a half step at the end
        ##VER1: p = p - np.array([t * epsilon for t in dU(x)])/2.

        # negate the momentum
        p= -p;
        current_U = U(current_x)
        current_K = K(current_p)
        #print(current_U)
        proposed_U = U(x)
        proposed_K = K(p)
        #C=np.exp(current_U-proposed_U+current_K-proposed_K)
        E_new = evalE(x)
        H_new = np.dot(p,p) / 2. + E_new
        dH = H - H_new
        #print("DH",dH)
        # accept/reject
        if np.random.rand() < np.exp(dH):
            current_x = x
            xall[j]=x
            accept+=1
            for tau in range(L):
                path.append(tmp[tau])
        else:
            xall[j] = current_x
        
        #H[j] = U(current_x)+K(current_p)
#    print("accept=",accept/np.double(N))
    return np.array(path), H, xall


# functions
U = lambda x: np.power(x,2)/2.
K = lambda p:  np.power(p,2)/2.
dU= lambda x: x


path, H, xall= HMC(U=U,K=K,dU=dU,N=100,x_0=[0,0], p_0=-10, epsilon=0.01, L=200)


def trajectory(pa,xa):
    #plt.plot([0],[0],'go')
    plt.plot(xa[:,0], xa[:,1], 'ko', label='HMC samples')
    plt.plot(pa[0:1000,0], pa[0:1000,1], 'r-', label='Trajectory (first 5 samples)')
    plt.legend()
    plt.show()

trajectory(path,xall)
