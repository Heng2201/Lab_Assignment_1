import matplotlib.pyplot as plt
import numpy as np
import os

def plot(epochs, pre_epoch, learning_rate, batch, Cost_list, Train_acc_list, Test_acc_list, save_dir : str):
             
    fig, ax = plt.subplots(figsize=(8,6))
    ax2 = ax.twinx()
    t = np.arange(0, epochs + pre_epoch)
    
    # plot the line
    lns1 = ax.plot(t,Cost_list, color = "blue", label = "cost")
    lns2 = ax2.plot(t,Train_acc_list, color = "red", label = "train_accuracy")
    lns3 = ax2.plot(t,Test_acc_list, color = "orange", label = "test_accuracy")

    # set title and label
    ax.set_title("Cost vs epochs with {} learning rate and batch size = {}".format(learning_rate, batch))
    ax.set_xlabel("epoch", fontsize = 14)
    ax.set_ylabel("cost", fontsize = 14)
    ax2.set_ylabel("accuracy", fontsize = 14)
                
    # from stack overflow, ways to combine all legends into one
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    
    # assign legend
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.75, pos.height])
    ax.legend(lns, labs, loc='upper right', bbox_to_anchor = (1.5, 1))

    Y = [ float(Train_acc_list[-1]), float(Test_acc_list[-1])]

    # show the value of the point (x,y)
    plt.annotate('({}, {:.2f}%)'.format(t[-1]+1, float(Train_acc_list[-1])) , xy = (t[-1] , float(Train_acc_list[-1])),  horizontalalignment = "right", verticalalignment = "bottom") 
    plt.annotate('test_acc:({}, {:.2f}%)'.format(t[-1]+1, float(Test_acc_list[-1])) , xy = (t[-1] , float(Test_acc_list[-1] )),\
                  xytext = (t[-1] , float(Test_acc_list[-1])),\
                  horizontalalignment = "right", verticalalignment = "top") 
    ax.annotate('({}, {:.2f}e-3)'.format(t[-1]+1, float(Cost_list[-1] * 1000)) , xy = (t[-1] , Cost_list[-1]), xytext = (t[-1], Cost_list[-1]),\
                  horizontalalignment = "right", verticalalignment = "bottom") 
    # arrowprops=dict(facecolor='black', shrink=0.05),

    try: 
        plt.savefig(save_dir + 'epochs_{}_batch_size_{}.png'.format(str(epochs + pre_epoch).zfill(3),str(batch)))

    except FileNotFoundError : 
        os.makedirs(save_dir)
        plt.savefig(save_dir + 'epochs_{}_batch_size_{}.png'.format(str(epochs + pre_epoch).zfill(3),str(batch)))
    plt.close("all")
    # plt.show()

def plot_random10(image, predict_result, actual_result):
   
    fig = plt.figure()

    ax1 = fig.add_subplot(2,5,1)
    ax1.imshow(image[:, 0].reshape(28,28), cmap = "gray")
    ax1.set_title("p: {}, a: {}".format(predict_result[0], actual_result[0]))
    ax2 = fig.add_subplot(2,5,2)
    ax2.imshow(image[:, 1].reshape(28,28), cmap = "gray")
    ax2.set_title("p: {}, a: {}".format(predict_result[1], actual_result[1]))
    ax3 = fig.add_subplot(2,5,3)
    ax3.imshow(image[:, 2].reshape(28,28), cmap = "gray")
    ax3.set_title("p: {}, a: {}".format(predict_result[2], actual_result[2]))
    ax4 = fig.add_subplot(2,5,4)
    ax4.imshow(image[:, 3].reshape(28,28), cmap = "gray")
    ax4.set_title("p: {}, a: {}".format(predict_result[3], actual_result[3]))
    ax5 = fig.add_subplot(2,5,5)
    ax5.imshow(image[:, 4].reshape(28,28), cmap = "gray")
    ax5.set_title("p: {}, a: {}".format(predict_result[4], actual_result[4]))
    ax6 = fig.add_subplot(2,5,6)
    ax6.imshow(image[:, 5].reshape(28,28), cmap = "gray")
    ax6.set_title("p: {}, a: {}".format(predict_result[5], actual_result[5]))
    ax7 = fig.add_subplot(2,5,7)
    ax7.imshow(image[:, 6].reshape(28,28), cmap = "gray")
    ax7.set_title("p: {}, a: {}".format(predict_result[6], actual_result[6]))
    ax8 = fig.add_subplot(2,5,8)
    ax8.imshow(image[:, 7].reshape(28,28), cmap = "gray")
    ax8.set_title("p: {}, a: {}".format(predict_result[7], actual_result[7]))
    ax9 = fig.add_subplot(2,5,9)
    ax9.imshow(image[:, 8].reshape(28,28), cmap = "gray")
    ax9.set_title("p: {}, a: {}".format(predict_result[8], actual_result[8]))
    ax10 = fig.add_subplot(2,5,10)
    ax10.imshow(image[:, 9].reshape(28,28), cmap = "gray")
    ax10.set_title("p: {}, a: {}".format(predict_result[9], actual_result[9]))

    fig.tight_layout()
    plt.show()
    
    
    
    
    pass
