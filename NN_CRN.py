import numpy as np
import utils
import arithmetic_CRNs as acrn

class NN_CRN_Leaky_ReLU : 
    def __init__(self, n1, n2, n3, alpha, beta, init_mu, init_sigma, dt, k_u, noise, timelen) : 
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        self.W1p = abs(np.random.normal(init_mu,init_sigma,n1*n2)).reshape([n2,n1])
        self.W1n = abs(np.random.normal(init_mu,init_sigma,n1*n2)).reshape([n2,n1])
        self.W2p = abs(np.random.normal(init_mu,init_sigma,n2*n3)).reshape([n3,n2])
        self.W2n = abs(np.random.normal(init_mu,init_sigma,n2*n3)).reshape([n3,n2])
        self.b1p = abs(np.random.normal(init_mu,init_sigma,n2))
        self.b1n = abs(np.random.normal(init_mu,init_sigma,n2))
        self.b2p = abs(np.random.normal(init_mu,init_sigma,n3))
        self.b2n = abs(np.random.normal(init_mu,init_sigma,n3))
        
        self.alpha = alpha
        self.beta = beta
        self.dt = dt 
        self.k_u = k_u
        self.noise = noise
        self.timelen = timelen
    
    def run(self, label_x, label_y, test_x, test_y, noise_controller, epoch, N, dataset_type):
        #noise controller) 0 = no noise / 1 = reaction rate noise for backward net / 2 = forward net layer noise / 3 = input noise per each new iteration
        train_loss = []
        validation_loss = []
        accuracy = []
        params = [[],[],[],[]]

        for k in range(N*epoch) : 
            #Accuracy test per 100 iteration
            if(k%100==0) : 
                tmp = 0
                for j in range(len(test_x)) : 
                    z1 = np.dot((self.W1p-self.W1n),test_x[j])+(self.b1p-self.b1n)
                    y1 = (np.abs(z1)+z1)/2*self.alpha + (-np.abs(z1)+z1)/2*self.beta
                    z2 = np.dot((self.W2p-self.W2n),y1)+(self.b2p-self.b2n)
                    y2 = (np.abs(z2)+z2)/2*self.alpha + (-np.abs(z2)+z2)/2*self.beta
                    if(dataset_type=='MNIST' or dataset_type=='iris') : 
                        tmp = tmp + (np.argmax(y2)==np.argmax(test_y[j])) #accuracy for MNIST, IRIS
                    else : 
                        tmp = tmp + (((0.5<y2))==test_y[j]) #accuracy for XOR
                accuracy.append(tmp/len(test_x))

            #Initialization
            index = np.random.randint(0,N)
            X = np.array([label_x[index]])
            Y_hat = np.array([label_y[index]])
            Y1p = np.array([np.zeros(self.n2)])+1
            Y1n =  np.array([np.zeros(self.n2)])+1
            Y2p = np.array([np.zeros(self.n3)])+1
            Y2n = np.array([np.zeros(self.n3)])+1
            temp1p = np.zeros((self.n3,self.n2))+1
            temp1n = np.zeros((self.n3,self.n2))+1
            temp2p = np.zeros(self.n2)+1
            temp2n = np.zeros(self.n2)+1
            temp4p = np.zeros((self.n2,self.n1))+1
            temp4n = np.zeros((self.n2,self.n1))+1
            temp5p = np.zeros(self.n3)+1
            temp5n = np.zeros(self.n3)+1
            temp6p = np.zeros(self.n2)+1
            temp6n = np.zeros(self.n2)+1

            if(noise_controller == 3) : X = X+np.random.normal(0,self.noise,np.shape(X))

            for j in range(self.timelen) : 
                if(j%10==0) : sigma = self.noise
                else : sigma = 0

                #forward
                dy1temp = (np.dot((self.W1p-self.W1n),X.T).T+(self.b1p-self.b1n))
                dy2temp = (np.dot((self.W2p-self.W2n),(Y1p-Y1n).T).T+(self.b2p-self.b2n))
                dy1pdt = (dy1temp>0)*dy1temp*self.alpha-Y1p
                dy1ndt = (dy1temp<0)*dy1temp*(-1)*self.beta-Y1n
                dy2pdt = (dy2temp>0)*dy2temp*self.alpha-Y2p
                dy2ndt = (dy2temp<0)*dy2temp*(-1)*self.beta-Y2n 
                if(noise_controller != 2) : 
                    Y1p=Y1p+dy1pdt*self.dt
                    Y1n=Y1n+dy1ndt*self.dt
                    Y2p=Y2p+dy2pdt*self.dt
                    Y2n=Y2n+dy2ndt*self.dt
                elif(noise_controller == 2) : 
                    Y1p=Y1p+dy1pdt*self.dt+np.random.normal(0,sigma,np.shape(Y1p))
                    Y1n=Y1n+dy1ndt*self.dt+np.random.normal(0,sigma,np.shape(Y1n))
                    Y2p=Y2p+dy2pdt*self.dt+np.random.normal(0,sigma,np.shape(Y2p))
                    Y2n=Y2n+dy2ndt*self.dt+np.random.normal(0,sigma,np.shape(Y2n))

                #backward
                if(noise_controller != 1) : 
                    #dL/dW2
                    dtemp1pdt = np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1n)+np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1p)
                    dtemp1ndt = np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1p)+np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1n)
                    
                    #dL/dy1
                    dtemp2pdt = (np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2p)+np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2n))[0]-temp2p
                    dtemp2ndt = (np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2n)+np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2p))[0]-temp2n
                    
                    #dL/dW1
                    dtemp4pdt = np.dot((np.multiply(temp2n,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta)).T,X)
                    dtemp4ndt = np.dot((np.multiply(temp2p,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta)).T,X)
                    
                    #gradient of b2
                    dtemp5pdt = (np.multiply(Y2n+Y_hat,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta))[0] 
                    dtemp5ndt = (np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta))[0]
                    
                    #gradient of b1
                    dtemp6pdt = (np.multiply(temp2n,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta))[0] 
                    dtemp6ndt = (np.multiply(temp2p,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta))[0]

                elif(noise_controller == 1) : 
                    #dL/dW2
                    dtemp1pdt = np.multiply(np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1n),np.random.normal(1,sigma,np.shape(self.W2p))/np.random.normal(1,sigma,np.shape(self.W2p)))+np.multiply(np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1p),np.random.normal(1,sigma,np.shape(self.W2p))/np.random.normal(1,sigma,np.shape(self.W2p)))
                    dtemp1ndt = np.multiply(np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1p),np.random.normal(1,sigma,np.shape(self.W2n))/np.random.normal(1,sigma,np.shape(self.W2n)))+np.multiply(np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1n),np.random.normal(1,sigma,np.shape(self.W2n))/np.random.normal(1,sigma,np.shape(self.W2n)))

                    #dL/dy1
                    dtemp2pdt = (np.multiply(np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2p),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2))+np.multiply(np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2n),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2)))[0]-np.multiply(temp2p,np.random.normal(1,sigma,self.n2))
                    dtemp2ndt = (np.multiply(np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2n),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2))+np.multiply(np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2p),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2)))[0]-np.multiply(temp2n,np.random.normal(1,sigma,self.n2))

                    #dL/dW1
                    dtemp4pdt = np.multiply(np.dot((np.multiply(temp2n,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta)).T,X),np.random.normal(1,sigma,np.shape(self.W1p))/np.random.normal(1,sigma,np.shape(self.W1p)))
                    dtemp4ndt = np.multiply(np.dot((np.multiply(temp2p,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta)).T,X),np.random.normal(1,sigma,np.shape(self.W1n))/np.random.normal(1,sigma,np.shape(self.W1n)))
                    
                    #gradient of b2
                    dtemp5pdt = np.multiply((np.multiply(Y2n+Y_hat,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta))[0],np.random.normal(1,sigma,np.shape(self.b2p))/np.random.normal(1,sigma,np.shape(self.b2p))) 
                    dtemp5ndt = np.multiply((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta))[0],np.random.normal(1,sigma,np.shape(self.b2n))/np.random.normal(1,sigma,np.shape(self.b2n)))
                    
                    #gradient of b1
                    dtemp6pdt = np.multiply((np.multiply(temp2n,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta))[0],np.random.normal(1,sigma,np.shape(self.b1p))/np.random.normal(1,sigma,np.shape(self.b1p))) 
                    dtemp6ndt = np.multiply((np.multiply(temp2p,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta))[0],np.random.normal(1,sigma,np.shape(self.b1n))/np.random.normal(1,sigma,np.shape(self.b1n))) 


                temp1p=temp1p+dtemp1pdt*self.dt
                temp2p=temp2p+dtemp2pdt*self.dt
                temp4p=temp4p+dtemp4pdt*self.dt
                temp5p=temp5p+dtemp5pdt*self.dt
                temp6p=temp6p+dtemp6pdt*self.dt
                temp1n=temp1n+dtemp1ndt*self.dt
                temp2n=temp2n+dtemp2ndt*self.dt
                temp4n=temp4n+dtemp4ndt*self.dt
                temp5n=temp5n+dtemp5ndt*self.dt
                temp6n=temp6n+dtemp6ndt*self.dt

                #update network(CRN)
                self.W2p=self.W2p+(dtemp1pdt)*self.dt*self.k_u
                self.W2n=self.W2n+(dtemp1ndt)*self.dt*self.k_u
                self.W1p=self.W1p+(dtemp4pdt)*self.dt*self.k_u
                self.W1n=self.W1n+(dtemp4ndt)*self.dt*self.k_u
                self.b2p=self.b2p+(dtemp5pdt)*self.dt*self.k_u
                self.b2n=self.b2n+(dtemp5ndt)*self.dt*self.k_u
                self.b1p=self.b1p+(dtemp6pdt)*self.dt*self.k_u
                self.b1n=self.b1n+(dtemp6ndt)*self.dt*self.k_u
                

            #Record train loss and parameters
            train_loss.append(utils.mean_error_for_leaky(self.W1p-self.W1n,self.W2p-self.W2n,self.b1p-self.b1n,self.b2p-self.b2n,label_x,label_y,self.alpha,self.beta))
            validation_loss.append(utils.mean_error_for_leaky(self.W1p-self.W1n,self.W2p-self.W2n,self.b1p-self.b1n,self.b2p-self.b2n,test_x,test_y,self.alpha,self.beta))
            params[0].append(self.W1p-self.W1n)
            params[1].append(self.W2p-self.W2n)
            params[2].append(self.b1p-self.b1n)
            params[3].append(self.b2p-self.b2n)
            print(k)

        return train_loss, validation_loss, accuracy, params
    
    def run_for_some_data(self, label_x, label_y, noise_controller, run_num, N):
        #noise controller) 0 = no noise / 1 = reaction rate noise for backward net / 2 = forward net layer noise / 3 = input noise per each new iteration
        Yvals = []
        w2vals = []
        DZ1vals = []
        DZ0vals = []

        for k in range(run_num) : 
            index = np.random.randint(0,N)
            X = np.array([label_x[index]])
            Y_hat = np.array([label_y[index]])
            Y1p = np.array([np.zeros(self.n2)])+1
            Y1n =  np.array([np.zeros(self.n2)])+1
            Y2p = np.array([np.zeros(self.n3)])+1
            Y2n = np.array([np.zeros(self.n3)])+1
            temp1p = np.zeros((self.n3,self.n2))+1
            temp1n = np.zeros((self.n3,self.n2))+1
            temp2p = np.zeros(self.n2)+1
            temp2n = np.zeros(self.n2)+1
            temp4p = np.zeros((self.n2,self.n1))+1
            temp4n = np.zeros((self.n2,self.n1))+1
            temp5p = np.zeros(self.n3)+1
            temp5n = np.zeros(self.n3)+1
            temp6p = np.zeros(self.n2)+1
            temp6n = np.zeros(self.n2)+1
            

            if(noise_controller == 3) : X = X+np.random.normal(0,self.noise,np.shape(X))

            for j in range(self.timelen) : 
                if(j%10==0) : sigma = self.noise
                else : sigma = 0

                #forward
                dy1temp = (np.dot((self.W1p-self.W1n),X.T).T+(self.b1p-self.b1n))
                dy2temp = (np.dot((self.W2p-self.W2n),(Y1p-Y1n).T).T+(self.b2p-self.b2n))
                dy1pdt = (dy1temp>0)*dy1temp*self.alpha-Y1p
                dy1ndt = (dy1temp<0)*dy1temp*(-1)*self.beta-Y1n
                dy2pdt = (dy2temp>0)*dy2temp*self.alpha-Y2p
                dy2ndt = (dy2temp<0)*dy2temp*(-1)*self.beta-Y2n 
                if(noise_controller != 2) : 
                    Y1p=Y1p+dy1pdt*self.dt
                    Y1n=Y1n+dy1ndt*self.dt
                    Y2p=Y2p+dy2pdt*self.dt
                    Y2n=Y2n+dy2ndt*self.dt
                elif(noise_controller == 2) : 
                    Y1p=Y1p+dy1pdt*self.dt+np.random.normal(0,sigma,np.shape(Y1p))
                    Y1n=Y1n+dy1ndt*self.dt+np.random.normal(0,sigma,np.shape(Y1n))
                    Y2p=Y2p+dy2pdt*self.dt+np.random.normal(0,sigma,np.shape(Y2p))
                    Y2n=Y2n+dy2ndt*self.dt+np.random.normal(0,sigma,np.shape(Y2n))
                
                w2vals.append((self.W2p-self.W2n)[0][0])
                Yvals.append((Y2p-Y2n)[0])
                DZ1vals.append((((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)[0][0])
                DZ0vals.append((((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta)[0][0])

                #backward
                if(noise_controller != 1) : 
                    #dL/dW2
                    dtemp1pdt = np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1n)+np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1p)
                    dtemp1ndt = np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1p)+np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1n)
                    
                    #dL/dy1
                    dtemp2pdt = (np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2p)+np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2n))[0]-temp2p
                    dtemp2ndt = (np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2n)+np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2p))[0]-temp2n
                    
                    #dL/dW1
                    dtemp4pdt = np.dot((np.multiply(temp2n,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta)).T,X)
                    dtemp4ndt = np.dot((np.multiply(temp2p,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta)).T,X)
                    
                    #gradient of b2
                    dtemp5pdt = (np.multiply(Y2n+Y_hat,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta))[0] 
                    dtemp5ndt = (np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta))[0]
                    
                    #gradient of b1
                    dtemp6pdt = (np.multiply(temp2n,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta))[0] 
                    dtemp6ndt = (np.multiply(temp2p,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta))[0]

                elif(noise_controller == 1) : 
                    #dL/dW2
                    dtemp1pdt = np.multiply(np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1n),np.random.normal(1,sigma,np.shape(self.W2p))/np.random.normal(1,sigma,np.shape(self.W2p)))+np.multiply(np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1p),np.random.normal(1,sigma,np.shape(self.W2p))/np.random.normal(1,sigma,np.shape(self.W2p)))
                    dtemp1ndt = np.multiply(np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1p),np.random.normal(1,sigma,np.shape(self.W2n))/np.random.normal(1,sigma,np.shape(self.W2n)))+np.multiply(np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)).T,Y1n),np.random.normal(1,sigma,np.shape(self.W2n))/np.random.normal(1,sigma,np.shape(self.W2n)))

                    #dL/dy1
                    dtemp2pdt = (np.multiply(np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2p),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2))+np.multiply(np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2n),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2)))[0]-np.multiply(temp2p,np.random.normal(1,sigma,self.n2))
                    dtemp2ndt = (np.multiply(np.dot((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2n),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2))+np.multiply(np.dot((np.multiply((Y2n+Y_hat),((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta)),self.W2p),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2)))[0]-np.multiply(temp2n,np.random.normal(1,sigma,self.n2))

                    #dL/dW1
                    dtemp4pdt = np.multiply(np.dot((np.multiply(temp2n,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta)).T,X),np.random.normal(1,sigma,np.shape(self.W1p))/np.random.normal(1,sigma,np.shape(self.W1p)))
                    dtemp4ndt = np.multiply(np.dot((np.multiply(temp2p,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta)).T,X),np.random.normal(1,sigma,np.shape(self.W1n))/np.random.normal(1,sigma,np.shape(self.W1n)))
                    
                    #gradient of b2
                    dtemp5pdt = np.multiply((np.multiply(Y2n+Y_hat,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta))[0],np.random.normal(1,sigma,np.shape(self.b2p))/np.random.normal(1,sigma,np.shape(self.b2p))) 
                    dtemp5ndt = np.multiply((np.multiply(Y2p,((Y2p-Y2n)>0)*self.alpha+((Y2p-Y2n)<=0)*self.beta))[0],np.random.normal(1,sigma,np.shape(self.b2n))/np.random.normal(1,sigma,np.shape(self.b2n)))
                    
                    #gradient of b1
                    dtemp6pdt = np.multiply((np.multiply(temp2n,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta))[0],np.random.normal(1,sigma,np.shape(self.b1p))/np.random.normal(1,sigma,np.shape(self.b1p))) 
                    dtemp6ndt = np.multiply((np.multiply(temp2p,((Y1p-Y1n)>0)*self.alpha+((Y1p-Y1n)<=0)*self.beta))[0],np.random.normal(1,sigma,np.shape(self.b1n))/np.random.normal(1,sigma,np.shape(self.b1n))) 


                temp1p=temp1p+dtemp1pdt*self.dt
                temp2p=temp2p+dtemp2pdt*self.dt
                temp4p=temp4p+dtemp4pdt*self.dt
                temp5p=temp5p+dtemp5pdt*self.dt
                temp6p=temp6p+dtemp6pdt*self.dt
                temp1n=temp1n+dtemp1ndt*self.dt
                temp2n=temp2n+dtemp2ndt*self.dt
                temp4n=temp4n+dtemp4ndt*self.dt
                temp5n=temp5n+dtemp5ndt*self.dt
                temp6n=temp6n+dtemp6ndt*self.dt

                #update network(CRN)
                self.W2p=self.W2p+(dtemp1pdt)*self.dt*self.k_u
                self.W2n=self.W2n+(dtemp1ndt)*self.dt*self.k_u
                self.W1p=self.W1p+(dtemp4pdt)*self.dt*self.k_u
                self.W1n=self.W1n+(dtemp4ndt)*self.dt*self.k_u
                self.b2p=self.b2p+(dtemp5pdt)*self.dt*self.k_u
                self.b2n=self.b2n+(dtemp5ndt)*self.dt*self.k_u
                self.b1p=self.b1p+(dtemp6pdt)*self.dt*self.k_u
                self.b1n=self.b1n+(dtemp6ndt)*self.dt*self.k_u

        return Yvals, w2vals, DZ1vals, DZ0vals

class NN_CRN_smoothed_ReLU : 
    def __init__(self, n1, n2, n3, H, init_mu, init_sigma, dt, k_u, noise, timelen) : 
        self.W1p = abs(np.random.normal(init_mu,init_sigma,n1*n2)).reshape([n2,n1])
        self.W1n = abs(np.random.normal(init_mu,init_sigma,n1*n2)).reshape([n2,n1])
        self.W2p = abs(np.random.normal(init_mu,init_sigma,n2*n3)).reshape([n3,n2])
        self.W2n = abs(np.random.normal(init_mu,init_sigma,n2*n3)).reshape([n3,n2])
        self.b1p = abs(np.random.normal(init_mu,init_sigma,n2))
        self.b1n = abs(np.random.normal(init_mu,init_sigma,n2))
        self.b2p = abs(np.random.normal(init_mu,init_sigma,n3))
        self.b2n = abs(np.random.normal(init_mu,init_sigma,n3))

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        self.H = H
        self.noise=noise
        self.dt = dt 
        self.k_u = k_u
        self.timelen = timelen
    
    def run(self, label_x, label_y, test_x, test_y, noise_controller, epoch, N, dataset_type) : 
        train_loss = []
        validation_loss = []
        accuracy = []
        params = [[],[],[],[]]

        for k in range(epoch*N) : 
            #Accuracy test per 100 iteration
            if(k%100==0) : 
                tmp = 0
                for j in range(len(test_x)) : 
                    z1 = np.dot((self.W1p-self.W1n),test_x[j])+(self.b1p-self.b1n)
                    y1 = (z1+np.sqrt(np.power(z1,2)+4*self.H))/2
                    z2 = np.dot((self.W2p-self.W2n),y1)+(self.b2p-self.b2n)
                    y2 = (z2+np.sqrt(np.power(z2,2)+4*self.H))/2
                    if(dataset_type=='MNIST' or dataset_type=='iris') : 
                        tmp = tmp + (np.argmax(y2)==np.argmax(test_y[j])) # accuracy for MNIST, IRIS
                    else : 
                        tmp = tmp + (((0.5<y2))==test_y[j]) # accuracy for XOR
                accuracy.append(tmp/len(test_x))

            #Initialization
            index = np.random.randint(0,N)
            X = np.array([label_x[index]])
            Y_hat = np.array([label_y[index]])

            Y1 = np.array([np.zeros(self.n2)])+1
            Y2 = np.array([np.zeros(self.n3)])+1
            temp0p = np.zeros(5*self.n3)
            temp0n = np.zeros(5*self.n3)
            temp1p = np.zeros((self.n3,self.n2))+1
            temp1n = np.zeros((self.n3,self.n2))+1
            temp2p = np.zeros(self.n2)+1
            temp2n = np.zeros(self.n2)+1
            temp3p = np.zeros(3*self.n2)
            temp3n = np.zeros(3*self.n2)
            temp4p = np.zeros((self.n2,self.n1))+1
            temp4n = np.zeros((self.n2,self.n1))+1
            temp5p = np.zeros(self.n3)+1
            temp5n = np.zeros(self.n3)+1
            temp6p = np.zeros(self.n2)+1
            temp6n = np.zeros(self.n2)+1
            
            if(noise_controller == 3) : X = X+np.random.normal(0,self.noise,np.shape(X))

            for j in range(self.timelen) :
                if(j%10==0) : sigma = self.noise
                else : sigma = 0
                
                #forward
                dy1dt = self.H+(np.dot((self.W1p-self.W1n),X.T).T+(self.b1p-self.b1n))*Y1-np.power(Y1,2)
                dy2dt = self.H+(np.dot((self.W2p-self.W2n),Y1.T).T+(self.b2p-self.b2n))*Y2-np.power(Y2,2)
                if(noise_controller != 2) : 
                    Y1=Y1+dy1dt*self.dt
                    Y2=Y2+dy2dt*self.dt
                elif(noise_controller == 2) : 
                    Y1=Y1+dy1dt*self.dt+np.random.normal(0,sigma,np.shape(Y1))
                    Y2=Y2+dy2dt*self.dt+np.random.normal(0,sigma,np.shape(Y2))

                #backward
                #dL/dz2
                if(noise_controller != 1) : 
                    dtemp0pdt = np.zeros(5*self.n3)
                    dtemp0ndt = np.zeros(5*self.n3)
                    for i in range(self.n3) : 
                        dtemp0pdt[i*5] = acrn.times(Y2[0,i],0,Y2[0,i],0,temp0p[i*5],temp0n[i*5])[0]
                        dtemp0pdt[i*5+1] = acrn.plus(temp0p[i*5],temp0n[i*5],self.H,0,temp0p[i*5+1],temp0n[i*5+1])[0]
                        dtemp0pdt[i*5+2] = acrn.divide(temp0p[i*5],temp0p[i*5+1],temp0p[i*5+2])[0]
                        dtemp0pdt[i*5+3] = acrn.minus(Y2[0,i],0,Y_hat[0,i],0,temp0p[i*5+3],temp0n[i*5+3])[0]
                        dtemp0pdt[i*5+4] = acrn.times(temp0p[i*5+2],temp0n[i*5+2],temp0p[i*5+3],temp0n[i*5+3],temp0p[i*5+4],temp0n[i*5+4])[0]

                        dtemp0ndt[i*5] = acrn.times(Y2[0,i],0,Y2[0,i],0,temp0p[i*5],temp0n[i*5])[1]
                        dtemp0ndt[i*5+1] = acrn.plus(temp0p[i*5],temp0n[i*5],self.H,0,temp0p[i*5+1],temp0n[i*5+1])[1]
                        dtemp0ndt[i*5+2] = 0
                        dtemp0ndt[i*5+3] = acrn.minus(Y2[0,i],0,Y_hat[0,i],0,temp0p[i*5+3],temp0n[i*5+3])[1]
                        dtemp0ndt[i*5+4] = acrn.times(temp0p[i*5+2],temp0n[i*5+2],temp0p[i*5+3],temp0n[i*5+3],temp0p[i*5+4],temp0n[i*5+4])[1]

                    #dL/dW2
                    dtemp1pdt = np.dot(np.array([[temp0n[i*5+4] for i in range(self.n3)]]).T,Y1)
                    dtemp1ndt = np.dot(np.array([[temp0p[i*5+4] for i in range(self.n3)]]).T,Y1)
                
                    #dL/dy1
                    dtemp2pdt = (np.dot(np.array([temp0p[i*5+4] for i in range(self.n3)]),self.W2p)+np.dot(np.array([temp0n[i*5+4] for i in range(self.n3)]),self.W2n))[0]-temp2p
                    dtemp2ndt = (np.dot(np.array([temp0n[i*5+4] for i in range(self.n3)]),self.W2p)+np.dot(np.array([temp0p[i*5+4] for i in range(self.n3)]),self.W2n))[0]-temp2n
                    
                    #dy1/dz1
                    dtemp3pdt = np.zeros(3*self.n2)
                    dtemp3ndt = np.zeros(3*self.n2)
                    for i in range(self.n2) : 
                        dtemp3pdt[i*3] = acrn.times(Y1[0,i],0,Y1[0,i],0,temp3p[i*3],temp3n[i*3])[0]
                        dtemp3pdt[i*3+1] = acrn.plus(temp3p[i*3],temp3n[i*3],self.H,0,temp3p[i*3+1],temp3n[i*3+1])[0]
                        dtemp3pdt[i*3+2] = acrn.divide(temp3p[i*3],temp3p[i*3+1],temp3p[i*3+2])[0]
                        dtemp3ndt[i*3] = acrn.times(Y1[0,i],0,Y1[0,i],0,temp3p[i*3],temp3n[i*3])[1]
                        dtemp3ndt[i*3+1] = acrn.plus(temp3p[i*3],temp3n[i*3],self.H,0,temp3p[i*3+1],temp3n[i*3+1])[1]
                        dtemp3ndt[i*3+2] = 0

                    #dL/dW1
                    dtemp4pdt = np.dot(np.array([np.multiply(temp2n,np.array([temp3p[i*3+2] for i in range(self.n2)]))+np.multiply(temp2p,np.array([temp3n[i*3+2] for i in range(self.n2)]))]).T,X)
                    dtemp4ndt = np.dot(np.array([np.multiply(temp2p,np.array([temp3p[i*3+2] for i in range(self.n2)]))+np.multiply(temp2n,np.array([temp3n[i*3+2] for i in range(self.n2)]))]).T,X)
                    
                    #gradient of b2
                    dtemp5pdt = np.array([temp0n[i*5+4] for i in range(self.n3)])
                    dtemp5ndt = np.array([temp0p[i*5+4] for i in range(self.n3)])
                    
                    #gradient of b1
                    dtemp6pdt = (np.multiply(temp2n,np.array([temp3p[i*3+2] for i in range(self.n2)])) + np.multiply(temp2p,np.array([temp3n[i*3+2] for i in range(self.n2)])))
                    dtemp6ndt = (np.multiply(temp2p,np.array([temp3p[i*3+2] for i in range(self.n2)])) + np.multiply(temp2n,np.array([temp3n[i*3+2] for i in range(self.n2)])))

                elif(noise_controller == 1) : 
                    dtemp0pdt = np.zeros(5*self.n3)
                    dtemp0ndt = np.zeros(5*self.n3)
                    for i in range(self.n3) : 
                        dtemp0pdt[i*5] = acrn.times(Y2[0,i],0,Y2[0,i],0,temp0p[i*5],temp0n[i*5])[0]
                        dtemp0pdt[i*5+1] = acrn.plus(temp0p[i*5],temp0n[i*5],self.H,0,temp0p[i*5+1],temp0n[i*5+1])[0]
                        dtemp0pdt[i*5+2] = acrn.divide(temp0p[i*5],temp0p[i*5+1],temp0p[i*5+2])[0]
                        dtemp0pdt[i*5+3] = acrn.minus(Y2[0,i],0,Y_hat[0,i],0,temp0p[i*5+3],temp0n[i*5+3])[0]
                        dtemp0pdt[i*5+4] = acrn.times(temp0p[i*5+2],temp0n[i*5+2],temp0p[i*5+3],temp0n[i*5+3],temp0p[i*5+4],temp0n[i*5+4])[0]

                        dtemp0ndt[i*5] = acrn.times(Y2[0,i],0,Y2[0,i],0,temp0p[i*5],temp0n[i*5])[1]
                        dtemp0ndt[i*5+1] = acrn.plus(temp0p[i*5],temp0n[i*5],self.H,0,temp0p[i*5+1],temp0n[i*5+1])[1]
                        dtemp0ndt[i*5+2] = 0
                        dtemp0ndt[i*5+3] = acrn.minus(Y2[0,i],0,Y_hat[0,i],0,temp0p[i*5+3],temp0n[i*5+3])[1]
                        dtemp0ndt[i*5+4] = acrn.times(temp0p[i*5+2],temp0n[i*5+2],temp0p[i*5+3],temp0n[i*5+3],temp0p[i*5+4],temp0n[i*5+4])[1]

                    #dL/dW2
                    dtemp1pdt = np.multiply(np.dot(np.array([[temp0n[i*5+4] for i in range(self.n3)]]).T,Y1),np.random.normal(1,sigma,[self.n3,self.n2])/np.random.normal(1,sigma,[self.n3,self.n2]))
                    dtemp1ndt = np.multiply(np.dot(np.array([[temp0p[i*5+4] for i in range(self.n3)]]).T,Y1),np.random.normal(1,sigma,[self.n3,self.n2])/np.random.normal(1,sigma,[self.n3,self.n2]))

                    #dL/dy1
                    dtemp2pdt = np.multiply((np.multiply(np.dot(np.array([temp0p[i*5+4] for i in range(self.n3)]),self.W2p),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2))+np.multiply(np.dot(np.array([temp0n[i*5+4] for i in range(self.n3)]),self.W2n),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2)))[0],np.random.normal(1,sigma,self.n2))-np.multiply(temp2p,np.random.normal(1,sigma,self.n2)) 
                    dtemp2ndt = np.multiply((np.multiply(np.dot(np.array([temp0n[i*5+4] for i in range(self.n3)]),self.W2p),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2))+np.multiply(np.dot(np.array([temp0p[i*5+4] for i in range(self.n3)]),self.W2n),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2)))[0],np.random.normal(1,sigma,self.n2))-np.multiply(temp2n,np.random.normal(1,sigma,self.n2))

                    #dy1/dz1
                    dtemp3pdt = np.zeros(3*self.n2)
                    dtemp3ndt = np.zeros(3*self.n2)
                    for i in range(self.n2) : 
                        dtemp3pdt[i*3] = acrn.times(Y1[0,i],0,Y1[0,i],0,temp3p[i*3],temp3n[i*3])[0]
                        dtemp3pdt[i*3+1] = acrn.plus(temp3p[i*3],temp3n[i*3],self.H,0,temp3p[i*3+1],temp3n[i*3+1])[0]
                        dtemp3pdt[i*3+2] = acrn.divide(temp3p[i*3],temp3p[i*3+1],temp3p[i*3+2])[0]
                        dtemp3ndt[i*3] = acrn.times(Y1[0,i],0,Y1[0,i],0,temp3p[i*3],temp3n[i*3])[1]
                        dtemp3ndt[i*3+1] = acrn.plus(temp3p[i*3],temp3n[i*3],self.H,0,temp3p[i*3+1],temp3n[i*3+1])[1]
                        dtemp3ndt[i*3+2] = 0

                    #dL/dW1
                    dtemp4pdt = np.multiply(np.dot(np.array([np.multiply(temp2n,np.array([temp3p[i*3+2] for i in range(self.n2)]))+np.multiply(temp2p,np.array([temp3n[i*3+2] for i in range(self.n2)]))]).T,X),np.random.normal(1,sigma,np.shape(self.W1p))/np.random.normal(1,sigma,np.shape(self.W1p)))
                    dtemp4ndt = np.multiply(np.dot(np.array([np.multiply(temp2p,np.array([temp3p[i*3+2] for i in range(self.n2)]))+np.multiply(temp2n,np.array([temp3n[i*3+2] for i in range(self.n2)]))]).T,X),np.random.normal(1,sigma,np.shape(self.W1p))/np.random.normal(1,sigma,np.shape(self.W1p)))

                    #gradient of b2
                    dtemp5pdt = np.multiply(([temp0n[i*5+4] for i in range(self.n3)]),np.random.normal(1,sigma,np.shape(self.b2p)))
                    dtemp5ndt = np.multiply(([temp0p[i*5+4] for i in range(self.n3)]),np.random.normal(1,sigma,np.shape(self.b2n)))
                    
                    #gradient of b1
                    dtemp6pdt = np.multiply((np.multiply(temp2n,np.array([temp3p[i*3+2] for i in range(self.n2)])) + np.multiply(temp2p,np.array([temp3n[i*3+2] for i in range(self.n2)]))),np.random.normal(1,sigma,np.shape(self.b1p))) 
                    dtemp6ndt = np.multiply((np.multiply(temp2p,np.array([temp3p[i*3+2] for i in range(self.n2)])) + np.multiply(temp2n,np.array([temp3n[i*3+2] for i in range(self.n2)]))),np.random.normal(1,sigma,np.shape(self.b1n)))
                
                temp0p=temp0p+dtemp0pdt*self.dt
                temp0n=temp0n+dtemp0ndt*self.dt
                temp1p=temp1p+dtemp1pdt*self.dt
                temp1n=temp1n+dtemp1ndt*self.dt
                temp2p=temp2p+dtemp2pdt*self.dt
                temp2n=temp2n+dtemp2ndt*self.dt
                temp3p=temp3p+dtemp3pdt*self.dt
                temp3n=temp3n+dtemp3ndt*self.dt
                temp4p=temp4p+dtemp4pdt*self.dt
                temp4n=temp4n+dtemp4ndt*self.dt
                temp5p=temp5p+dtemp5pdt*self.dt
                temp5n=temp5n+dtemp5ndt*self.dt
                temp6p=temp6p+dtemp6pdt*self.dt
                temp6n=temp6n+dtemp6ndt*self.dt

                #update network(CRN)
                self.W2p=self.W2p+(dtemp1pdt)*self.dt*self.k_u
                self.W2n=self.W2n+(dtemp1ndt)*self.dt*self.k_u
                self.W1p=self.W1p+(dtemp4pdt)*self.dt*self.k_u
                self.W1n=self.W1n+(dtemp4ndt)*self.dt*self.k_u
                self.b2p=self.b2p+(dtemp5pdt)*self.dt*self.k_u
                self.b2n=self.b2n+(dtemp5ndt)*self.dt*self.k_u
                self.b1p=self.b1p+(dtemp6pdt)*self.dt*self.k_u
                self.b1n=self.b1n+(dtemp6ndt)*self.dt*self.k_u

            #Record train loss and parameters
            train_loss.append(utils.mean_error(self.W1p-self.W1n,self.W2p-self.W2n,self.b1p-self.b1n,self.b2p-self.b2n,label_x,label_y,self.H))
            validation_loss.append(utils.mean_error(self.W1p-self.W1n,self.W2p-self.W2n,self.b1p-self.b1n,self.b2p-self.b2n,test_x,test_y,self.H))
            params[0].append(self.W1p-self.W1n)
            params[1].append(self.W2p-self.W2n)
            params[2].append(self.b1p-self.b1n)
            params[3].append(self.b2p-self.b2n)
            print(k)
        
        return train_loss, validation_loss, accuracy, params
    
    def run_for_some_data(self, label_x, label_y, noise_controller, run_num, N) : 
        Yvals = []
        w2vals = []
        DZ1vals = []
        DZ0vals = []

        for k in range(run_num) : 
            index = np.random.randint(0,N)
            X = np.array([label_x[index]])
            Y_hat = np.array([label_y[index]])

            Y1 = np.array([np.zeros(self.n2)])+1
            Y2 = np.array([np.zeros(self.n3)])+1
            temp0p = np.zeros(5*self.n3)
            temp0n = np.zeros(5*self.n3)
            temp1p = np.zeros((self.n3,self.n2))+1
            temp1n = np.zeros((self.n3,self.n2))+1
            temp2p = np.zeros(self.n2)+1
            temp2n = np.zeros(self.n2)+1
            temp3p = np.zeros(3*self.n2)
            temp3n = np.zeros(3*self.n2)
            temp4p = np.zeros((self.n2,self.n1))+1
            temp4n = np.zeros((self.n2,self.n1))+1
            temp5p = np.zeros(self.n3)+1
            temp5n = np.zeros(self.n3)+1
            temp6p = np.zeros(self.n2)+1
            temp6n = np.zeros(self.n2)+1
            
            
            if(noise_controller == 3) : X = X+np.random.normal(0,self.noise,np.shape(X))

            for j in range(self.timelen) :
                if(j%10==0) : sigma = self.noise
                else : sigma = 0
                
                #forward
                dy1dt = self.H+(np.dot((self.W1p-self.W1n),X.T).T+(self.b1p-self.b1n))*Y1-np.power(Y1,2)
                dy2dt = self.H+(np.dot((self.W2p-self.W2n),Y1.T).T+(self.b2p-self.b2n))*Y2-np.power(Y2,2)
                if(noise_controller != 2) : 
                    Y1=Y1+dy1dt*self.dt
                    Y2=Y2+dy2dt*self.dt
                elif(noise_controller == 2) : 
                    Y1=Y1+dy1dt*self.dt+np.random.normal(0,sigma,np.shape(Y1))
                    Y2=Y2+dy2dt*self.dt+np.random.normal(0,sigma,np.shape(Y2))

                w2vals.append((self.W2p-self.W2n)[0][0])
                Yvals.append(Y2[0])
                DZ1vals.append((temp0p-temp0n)[4])
                DZ0vals.append((temp3p-temp3n)[2])

                #backward
                #dL/dz2
                if(noise_controller != 1) : 
                    dtemp0pdt = np.zeros(5*self.n3)
                    dtemp0ndt = np.zeros(5*self.n3)
                    for i in range(self.n3) : 
                        dtemp0pdt[i*5] = acrn.times(Y2[0,i],0,Y2[0,i],0,temp0p[i*5],temp0n[i*5])[0]
                        dtemp0pdt[i*5+1] = acrn.plus(temp0p[i*5],temp0n[i*5],self.H,0,temp0p[i*5+1],temp0n[i*5+1])[0]
                        dtemp0pdt[i*5+2] = acrn.divide(temp0p[i*5],temp0p[i*5+1],temp0p[i*5+2])[0]
                        dtemp0pdt[i*5+3] = acrn.minus(Y2[0,i],0,Y_hat[0,i],0,temp0p[i*5+3],temp0n[i*5+3])[0]
                        dtemp0pdt[i*5+4] = acrn.times(temp0p[i*5+2],temp0n[i*5+2],temp0p[i*5+3],temp0n[i*5+3],temp0p[i*5+4],temp0n[i*5+4])[0]

                        dtemp0ndt[i*5] = acrn.times(Y2[0,i],0,Y2[0,i],0,temp0p[i*5],temp0n[i*5])[1]
                        dtemp0ndt[i*5+1] = acrn.plus(temp0p[i*5],temp0n[i*5],self.H,0,temp0p[i*5+1],temp0n[i*5+1])[1]
                        dtemp0ndt[i*5+2] = 0
                        dtemp0ndt[i*5+3] = acrn.minus(Y2[0,i],0,Y_hat[0,i],0,temp0p[i*5+3],temp0n[i*5+3])[1]
                        dtemp0ndt[i*5+4] = acrn.times(temp0p[i*5+2],temp0n[i*5+2],temp0p[i*5+3],temp0n[i*5+3],temp0p[i*5+4],temp0n[i*5+4])[1]

                    #dL/dW2
                    dtemp1pdt = np.dot(np.array([[temp0n[i*5+4] for i in range(self.n3)]]).T,Y1)
                    dtemp1ndt = np.dot(np.array([[temp0p[i*5+4] for i in range(self.n3)]]).T,Y1)
                
                    #dL/dy1
                    dtemp2pdt = (np.dot(np.array([temp0p[i*5+4] for i in range(self.n3)]),self.W2p)+np.dot(np.array([temp0n[i*5+4] for i in range(self.n3)]),self.W2n))[0]-temp2p
                    dtemp2ndt = (np.dot(np.array([temp0n[i*5+4] for i in range(self.n3)]),self.W2p)+np.dot(np.array([temp0p[i*5+4] for i in range(self.n3)]),self.W2n))[0]-temp2n
                    
                    #dy1/dz1
                    dtemp3pdt = np.zeros(3*self.n2)
                    dtemp3ndt = np.zeros(3*self.n2)
                    for i in range(self.n2) : 
                        dtemp3pdt[i*3] = acrn.times(Y1[0,i],0,Y1[0,i],0,temp3p[i*3],temp3n[i*3])[0]
                        dtemp3pdt[i*3+1] = acrn.plus(temp3p[i*3],temp3n[i*3],self.H,0,temp3p[i*3+1],temp3n[i*3+1])[0]
                        dtemp3pdt[i*3+2] = acrn.divide(temp3p[i*3],temp3p[i*3+1],temp3p[i*3+2])[0]
                        dtemp3ndt[i*3] = acrn.times(Y1[0,i],0,Y1[0,i],0,temp3p[i*3],temp3n[i*3])[1]
                        dtemp3ndt[i*3+1] = acrn.plus(temp3p[i*3],temp3n[i*3],self.H,0,temp3p[i*3+1],temp3n[i*3+1])[1]
                        dtemp3ndt[i*3+2] = 0

                    #dL/dW1
                    dtemp4pdt = np.dot(np.array([np.multiply(temp2n,np.array([temp3p[i*3+2] for i in range(self.n2)]))+np.multiply(temp2p,np.array([temp3n[i*3+2] for i in range(self.n2)]))]).T,X)
                    dtemp4ndt = np.dot(np.array([np.multiply(temp2p,np.array([temp3p[i*3+2] for i in range(self.n2)]))+np.multiply(temp2n,np.array([temp3n[i*3+2] for i in range(self.n2)]))]).T,X)
                    
                    #gradient of b2
                    dtemp5pdt = np.array([temp0n[i*5+4] for i in range(self.n3)])
                    dtemp5ndt = np.array([temp0p[i*5+4] for i in range(self.n3)])
                    
                    #gradient of b1
                    dtemp6pdt = (np.multiply(temp2n,np.array([temp3p[i*3+2] for i in range(self.n2)])) + np.multiply(temp2p,np.array([temp3n[i*3+2] for i in range(self.n2)])))
                    dtemp6ndt = (np.multiply(temp2p,np.array([temp3p[i*3+2] for i in range(self.n2)])) + np.multiply(temp2n,np.array([temp3n[i*3+2] for i in range(self.n2)])))

                elif(noise_controller == 1) : 
                    dtemp0pdt = np.zeros(5*self.n3)
                    dtemp0ndt = np.zeros(5*self.n3)
                    for i in range(self.n3) : 
                        dtemp0pdt[i*5] = acrn.times(Y2[0,i],0,Y2[0,i],0,temp0p[i*5],temp0n[i*5])[0]
                        dtemp0pdt[i*5+1] = acrn.plus(temp0p[i*5],temp0n[i*5],self.H,0,temp0p[i*5+1],temp0n[i*5+1])[0]
                        dtemp0pdt[i*5+2] = acrn.divide(temp0p[i*5],temp0p[i*5+1],temp0p[i*5+2])[0]
                        dtemp0pdt[i*5+3] = acrn.minus(Y2[0,i],0,Y_hat[0,i],0,temp0p[i*5+3],temp0n[i*5+3])[0]
                        dtemp0pdt[i*5+4] = acrn.times(temp0p[i*5+2],temp0n[i*5+2],temp0p[i*5+3],temp0n[i*5+3],temp0p[i*5+4],temp0n[i*5+4])[0]

                        dtemp0ndt[i*5] = acrn.times(Y2[0,i],0,Y2[0,i],0,temp0p[i*5],temp0n[i*5])[1]
                        dtemp0ndt[i*5+1] = acrn.plus(temp0p[i*5],temp0n[i*5],self.H,0,temp0p[i*5+1],temp0n[i*5+1])[1]
                        dtemp0ndt[i*5+2] = 0
                        dtemp0ndt[i*5+3] = acrn.minus(Y2[0,i],0,Y_hat[0,i],0,temp0p[i*5+3],temp0n[i*5+3])[1]
                        dtemp0ndt[i*5+4] = acrn.times(temp0p[i*5+2],temp0n[i*5+2],temp0p[i*5+3],temp0n[i*5+3],temp0p[i*5+4],temp0n[i*5+4])[1]

                    #dL/dW2
                    dtemp1pdt = np.multiply(np.dot(np.array([[temp0n[i*5+4] for i in range(self.n3)]]).T,Y1),np.random.normal(1,sigma,[self.n3,self.n2])/np.random.normal(1,sigma,[self.n3,self.n2]))
                    dtemp1ndt = np.multiply(np.dot(np.array([[temp0p[i*5+4] for i in range(self.n3)]]).T,Y1),np.random.normal(1,sigma,[self.n3,self.n2])/np.random.normal(1,sigma,[self.n3,self.n2]))

                    #dL/dy1
                    dtemp2pdt = np.multiply((np.multiply(np.dot(np.array([temp0p[i*5+4] for i in range(self.n3)]),self.W2p),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2))+np.multiply(np.dot(np.array([temp0n[i*5+4] for i in range(self.n3)]),self.W2n),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2)))[0],np.random.normal(1,sigma,self.n2))-np.multiply(temp2p,np.random.normal(1,sigma,self.n2)) 
                    dtemp2ndt = np.multiply((np.multiply(np.dot(np.array([temp0n[i*5+4] for i in range(self.n3)]),self.W2p),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2))+np.multiply(np.dot(np.array([temp0p[i*5+4] for i in range(self.n3)]),self.W2n),np.random.normal(1,sigma,self.n2)/np.random.normal(1,sigma,self.n2)))[0],np.random.normal(1,sigma,self.n2))-np.multiply(temp2n,np.random.normal(1,sigma,self.n2))

                    #dy1/dz1
                    dtemp3pdt = np.zeros(3*self.n2)
                    dtemp3ndt = np.zeros(3*self.n2)
                    for i in range(self.n2) : 
                        dtemp3pdt[i*3] = acrn.times(Y1[0,i],0,Y1[0,i],0,temp3p[i*3],temp3n[i*3])[0]
                        dtemp3pdt[i*3+1] = acrn.plus(temp3p[i*3],temp3n[i*3],self.H,0,temp3p[i*3+1],temp3n[i*3+1])[0]
                        dtemp3pdt[i*3+2] = acrn.divide(temp3p[i*3],temp3p[i*3+1],temp3p[i*3+2])[0]
                        dtemp3ndt[i*3] = acrn.times(Y1[0,i],0,Y1[0,i],0,temp3p[i*3],temp3n[i*3])[1]
                        dtemp3ndt[i*3+1] = acrn.plus(temp3p[i*3],temp3n[i*3],self.H,0,temp3p[i*3+1],temp3n[i*3+1])[1]
                        dtemp3ndt[i*3+2] = 0

                    #dL/dW1
                    dtemp4pdt = np.multiply(np.dot(np.array([np.multiply(temp2n,np.array([temp3p[i*3+2] for i in range(self.n2)]))+np.multiply(temp2p,np.array([temp3n[i*3+2] for i in range(self.n2)]))]).T,X),np.random.normal(1,sigma,np.shape(self.W1p))/np.random.normal(1,sigma,np.shape(self.W1p)))
                    dtemp4ndt = np.multiply(np.dot(np.array([np.multiply(temp2p,np.array([temp3p[i*3+2] for i in range(self.n2)]))+np.multiply(temp2n,np.array([temp3n[i*3+2] for i in range(self.n2)]))]).T,X),np.random.normal(1,sigma,np.shape(self.W1p))/np.random.normal(1,sigma,np.shape(self.W1p)))

                    #gradient of b2
                    dtemp5pdt = np.multiply(([temp0n[i*5+4] for i in range(self.n3)]),np.random.normal(1,sigma,np.shape(self.b2p)))
                    dtemp5ndt = np.multiply(([temp0p[i*5+4] for i in range(self.n3)]),np.random.normal(1,sigma,np.shape(self.b2n)))
                    
                    #gradient of b1
                    dtemp6pdt = np.multiply((np.multiply(temp2n,np.array([temp3p[i*3+2] for i in range(self.n2)])) + np.multiply(temp2p,np.array([temp3n[i*3+2] for i in range(self.n2)]))),np.random.normal(1,sigma,np.shape(self.b1p))) 
                    dtemp6ndt = np.multiply((np.multiply(temp2p,np.array([temp3p[i*3+2] for i in range(self.n2)])) + np.multiply(temp2n,np.array([temp3n[i*3+2] for i in range(self.n2)]))),np.random.normal(1,sigma,np.shape(self.b1n)))
                
                temp0p=temp0p+dtemp0pdt*self.dt
                temp0n=temp0n+dtemp0ndt*self.dt
                temp1p=temp1p+dtemp1pdt*self.dt
                temp1n=temp1n+dtemp1ndt*self.dt
                temp2p=temp2p+dtemp2pdt*self.dt
                temp2n=temp2n+dtemp2ndt*self.dt
                temp3p=temp3p+dtemp3pdt*self.dt
                temp3n=temp3n+dtemp3ndt*self.dt
                temp4p=temp4p+dtemp4pdt*self.dt
                temp4n=temp4n+dtemp4ndt*self.dt
                temp5p=temp5p+dtemp5pdt*self.dt
                temp5n=temp5n+dtemp5ndt*self.dt
                temp6p=temp6p+dtemp6pdt*self.dt
                temp6n=temp6n+dtemp6ndt*self.dt

                #update network(CRN)
                self.W2p=self.W2p+(dtemp1pdt)*self.dt*self.k_u
                self.W2n=self.W2n+(dtemp1ndt)*self.dt*self.k_u
                self.W1p=self.W1p+(dtemp4pdt)*self.dt*self.k_u
                self.W1n=self.W1n+(dtemp4ndt)*self.dt*self.k_u
                self.b2p=self.b2p+(dtemp5pdt)*self.dt*self.k_u
                self.b2n=self.b2n+(dtemp5ndt)*self.dt*self.k_u
                self.b1p=self.b1p+(dtemp6pdt)*self.dt*self.k_u
                self.b1n=self.b1n+(dtemp6ndt)*self.dt*self.k_u

        return Yvals, w2vals, DZ1vals, DZ0vals

        
    