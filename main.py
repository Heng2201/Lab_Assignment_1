import numpy as np
import os
import argparse

import func
import obj
import plot

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 0, help = "Number of epochs.")
    parser.add_argument('--pre_epoch', type = int, default = 60, help = "previous epochs that done")  
    parser.add_argument('--lr', type = int, default = 0.4, help = "Learning rate")
    parser.add_argument('--save_interval', type = int, default = 10, help = "interval of saving the plot")
    parser.add_argument('--batch_size', type = int, default = 10, help = "batch_size")  
    parser.add_argument('--data_dir', type = str, default = "mnist.npz", help = "directory of the dataset in this workspace")
    parser.add_argument('--pre_dir', type = str, default = "./params/100neurons/dif_lr/lr_0.4/", help = "directory of previous parameters")
    parser.add_argument('--param_dir', type = str, default = "./params/100neurons/dif_lr/lr_0.4/", help = "directory to save previous parameters")
    parser.add_argument('--plt_dir', type = str, default = "./fig/100neurons/dif_lr/lr_0.4/", help = "directory to save plot")
    parser.add_argument('--guess_10', type = bool, default = False, help = "Guess random 10")

    opt = parser.parse_args()

    # os.getcwd() is to get the current workspace directory
    path = os.getcwd()

    # os.path.join() combines two directories
    data = np.load(os.path.join(path, opt.data_dir))

    # transpose the image array is for calculation convenient 
    # x_train (pixels, # of pic)
    x_train, y_train = func.resize_img(data["x_train"]).T, func.one_hot(data["y_train"]).T
    x_test, y_test = func.resize_img(data["x_test"]).T, func.one_hot(data["y_test"]).T
    
    data.close()
    
    # print("x_train: {}".format(x_train.shape) + " y_train: {}".format(y_train.shape))

    try : 
        datas = dict(np.load(opt.pre_dir + "params_e{}.npz".format( opt.pre_epoch)))
        history = dict(np.load(opt.pre_dir + "history_e{}.npz".format(opt.pre_epoch)))
        parameter = datas
        cost_hist = history["cost_hist"].tolist()
        train_acc_hist = history["train_acc_hist"].tolist()
        test_acc_hist = history["test_acc_hist"].tolist()
        print("File open : batch size = {}".format(opt.batch_size))
        # print(parameter.keys())
        
        
    except FileNotFoundError : 
        parameter = None
        cost_hist = []
        train_acc_hist = []
        test_acc_hist = []
        print("No file found")
      
    # normal 500 300
    # complex 1000 600
    a = obj.NN(x_train.shape[0], 100, 150, y_train.shape[0],parameter)
    

    cost_list, train_acc_list, test_acc_list = a.train_and_test(x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy(), opt.lr, opt.epochs, opt.batch_size, opt.pre_epoch, opt.save_interval, \
                                                                cost_hist, train_acc_hist, test_acc_hist, opt.param_dir, opt.plt_dir)
    # plot.plot(opt.epochs, opt.pre_epoch, opt.lr, opt.batch_size, cost_list, train_acc_list, test_acc_list, opt.plt_dir)
    if opt.guess_10 == True : a.guess_10(x_test.copy(),y_test.copy()) 
    a.see_error(x_test.copy(),y_test.copy()) 
    try:
        datas.clear()
        history.clear()
    
    except NameError:
        pass
    
        
    