import numpy as np
import os

import plot

class NN:
    def __init__(self,inl, hl1, hl2 , outl, parameters = None):
        self.structure = self.__dim_struc(inl, hl1, hl2, outl)
        if type(parameters) == type(None) : self.parameters = self.__dim_param()
        else : 
            self.parameters = parameters
            print(self.parameters.keys())

    # declare the structure of the model 
    def __dim_struc(self, inl, hl1, hl2, outl):
        structure = [{"input" : inl, "output" : hl1, "activation": "relu"},\
                     {"input" : hl1, "output" : hl2, "activation": "relu"},\
                     {"input" : hl2, "output" : outl, "activation": "softmax"}]
        
        return structure

    # define all those weight and bias that need to be used in our model
    def __dim_param(self):
        params = {}

        for idx, element in enumerate(self.structure):
            
            # make the random predictable  
            np.random.seed(0)
            params["w" + str(idx + 1)] = np.random.rand(element["output"], element["input"]) * 0.01 
            np.random.seed(0)
            params["b" + str(idx + 1)] = np.random.rand(element["output"], 1) * 0.01 
        
        return params

    # function to implement single layer front propagation 
    def __onelayer_frontpropagation(self,A_prev, weight, bias,activation):

        Z_curr = np.dot(weight, A_prev) + bias
        if activation == "relu": A_curr = self.__relu(Z_curr)
        elif activation == "softmax": A_curr = self.__softmax(Z_curr)
        
        return A_curr, Z_curr

    def __onelayer_backpropagation(self, dZ_next, w_next, A_curr, A_prev, activation, m, y):

        NoneType = type(None)

        # eval() evaluate string to callable function name
        if activation == "softmax": df = self.__softmax_backward(A_curr, y)
        elif activation == "relu" : df = self.__relu_backward(A_curr)


        if type(w_next) == NoneType and type(dZ_next) == NoneType : dZ_curr = df
        else : dZ_curr = (1 / m) * np.dot(w_next.T, dZ_next) * df
        try:dW_curr = (1 / m) * np.dot(dZ_curr, A_prev.T)
        except ZeroDivisionError : print("m: {}".format(m))
        db_curr = (1 / m) * np.sum(dZ_curr, axis = 1, keepdims = True)

        return dZ_curr, dW_curr, db_curr

    # function did the whole back propagation and return the gradients of each parameters
    def __backpropagation(self, memory, y):
        gradients = {}

        # because list is a iterable object, so if not using copy it will change the original list
        reverse_struc = self.structure.copy()
        reverse_struc.reverse()
        m = memory["A0"].shape[1]
        w_next = None
        dZ_next = None

        for idx, element in enumerate(reverse_struc):
            activation = element["activation"]
            A_prev = memory["A" + str(2 - idx)]
            A_curr = memory["A" + str(3 - idx)]

            dZ_curr, dW_curr, db_curr = self.__onelayer_backpropagation(dZ_next = dZ_next, w_next = w_next, A_curr = A_curr,\
                                                                        A_prev = A_prev, activation = activation, m = m, y = y)
            
            gradients["dw" + str(3 - idx)] = dW_curr
            gradients["db" + str(3 - idx)] = db_curr

            dZ_next = dZ_curr
            w_next = self.parameters["w" + str(3 - idx)]

        return gradients
    
    def __update_params(self, gradients, learning_rate):

        for idx, _ in enumerate(self.structure):
            self.parameters["w" + str(idx + 1)] = self.parameters["w" + str(idx + 1)] - learning_rate * gradients["dw" + str(idx + 1)]
            self.parameters["b" + str(idx + 1)] = self.parameters["b" + str(idx + 1)] - learning_rate * gradients["db" + str(idx + 1)]


    def __relu(self, x):
        return np.maximum(0,x)
    
    def __softmax(self,x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, 0)
 
    def __relu_backward(self,dx):
        return np.array(dx > 0, dtype = np.float32)
    
    def __softmax_backward(self, dA, y):
        return (dA - y)
    

    # function that do whole forward propagation 
    # this function will also define the data of input layer as A0
    def forwardpropagation(self, input):
        memory = {}
        A_curr = input / 255
        memory["A0"] = input / 255

        for idx, element in enumerate(self.structure):
            A_prev = A_curr

            activation = element["activation"]
            w_curr = self.parameters["w" + str(idx + 1)]
            b_curr = self.parameters["b" + str(idx + 1)]
            A_curr, Z_curr = self.__onelayer_frontpropagation(A_prev, w_curr, b_curr, activation)
            
            memory["A" + str(idx + 1)] = A_curr
            memory["Z" + str(idx + 1)] = Z_curr


        return A_curr, memory

    # calculate cost
    def get_cost(self,y_hat,y):
        m = y_hat.shape[1]

        # np.sum sum all the value in the array together if axis = None
        return - (1 / m) * np.sum(y * np.log(y_hat))
    

    def get_accuracy(self, x, y, dataset : str):

        y_hat , _= self.forwardpropagation(x)

        y_hat_arg = np.argmax(y_hat, 0)
        y_arg = np.argmax(y, 0)

        acc = np.mean(y_hat_arg == y_arg, 0) * 100

        return acc, y_hat, y
    
    def train_and_test(self, x, y, x_test, y_test, learning_rate : float, epochs : int, batch_size : int,pre_epochs : int, save_interval : int, cost_hist : list,\
                       train_acc_hist : list, test_acc_hist : list, save_dir : str, plt_dir : str):
        
        
        cost_history = cost_hist
        train_accurate_history = train_acc_hist
        test_accurate_history = test_acc_hist

        if batch_size == None : batch_size = x.shape[1]
        else : pass

        for i in range(epochs):
            
            if batch_size == x.shape[1]:
                
                y_hat, memory = self.forwardpropagation(x.copy)
                
                gradients = self.__backpropagation(memory = memory, y = y)
                self.__update_params(gradients, learning_rate)
            
            else:
                for j in range(int(x.shape[1] / batch_size) + 1):
                    
                    x_batch = x[: , batch_size * j : batch_size * (j + 1)].copy()
                    y_batch = y[: , batch_size * j : batch_size * (j + 1)].copy()
                    if x_batch.shape[1] == 0 : continue
                    y_hat, memory = self.forwardpropagation(x_batch.copy())
                    gradients = self.__backpropagation(memory, y_batch.copy())
                    self.__update_params(gradients, learning_rate)

            

            # get the lost of this epochs
            train_acc, y_hat, _ = self.get_accuracy(x.copy(), y.copy(), dataset = "Training")
            cost = self.get_cost(y_hat.copy(), y.copy())
            test_acc, y_test_hat, _ = self.get_accuracy(x_test.copy(), y_test.copy(), dataset = "Testing")
            
            cost_history.append(cost)
            train_accurate_history.append(train_acc)
            test_accurate_history.append(test_acc)
            if (i + 1) % save_interval == 0:
                history = {"cost_hist" : cost_history, "train_acc_hist" : train_accurate_history, "test_acc_hist" : test_accurate_history}

                try: 
                    np.savez(save_dir +"params_e{}".format(pre_epochs + i + 1), **self.parameters)
                    np.savez(save_dir + "history_e{}".format(pre_epochs + i + 1), **history)

                except FileNotFoundError : 
                    os.makedirs(save_dir)
                    np.savez(save_dir +"params_e{}".format(pre_epochs + i + 1), **self.parameters)
                    np.savez(save_dir + "history_e{}".format(pre_epochs + i + 1), **history)

                plot.plot(i + 1, pre_epochs, learning_rate, batch_size, cost_hist, train_accurate_history, test_accurate_history, plt_dir)

            if (pre_epochs + i + 1) % 10 == 0:
                print("epoch {} done".format(pre_epochs + i + 1))
            

        return cost_history, train_accurate_history, test_accurate_history


    def guess_10(self, x, y):
        
        random = np.random.randint(x.shape[1], size =10)
        img_10 = x.copy()[: , random]        
        y_hat, _ = self.forwardpropagation(img_10.copy())
        predict_result = np.argmax(y_hat.copy(), 0)
        actual_result = np.argmax(y.copy()[:, random], 0)
        plot.plot_random10(image = img_10, predict_result = predict_result, actual_result = actual_result)
 
        
    def see_error(self, x, y):
        y_hat, _ = self.forwardpropagation(x.copy())
        predict_result = np.argmax(y_hat.copy(), 0)
        actual_result = np.argmax(y.copy(), 0)

        # return the index number of which predict_result != actual_result
        error_idx = np.where(predict_result != actual_result)[0]
        print("The model predict {} wrongly".format(error_idx.shape[0]))
        
        random = np.random.randint(0, error_idx.shape[0], size = 10)
        random_10_error_idx = error_idx[random]
        error_10 = predict_result[random_10_error_idx]
        image_10 = x.copy()[: , random_10_error_idx]
        actual_10 = actual_result[random_10_error_idx]

    
        plot.plot_random10(image = image_10, predict_result = error_10, actual_result = actual_10)


        pass




