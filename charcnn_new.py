# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 22:05:49 2021

@author: Asus
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 22:40:36 2021
@author: Asus
"""

import numpy as np
import theano
import theano.tensor as T

from sklearn.model_selection import train_test_split
from theano.tensor.nnet import conv
from theano.tensor.signal import pool as downsample


from architecture import ConvolutionalLayer
from architecture import EmbedIDLayer
from architecture import FullyConnectedLayer
from architecture import MaxPoolingLayer

from optimizer import *
from utility import *
from loader import *

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4

import tkinter as tk

class CharSCNN(object):
    def __init__(
            self,
            rng,
            batchsize=100,
            activation=relu,
    ):
            
        (charmap, wordsmap, numsent, charcnt, wordcnt, maxwordlen, maxsenlen, kchr, kwrd, xchr, xwrd, y)= read("dataset_good_preprocessed.txt")
        # (numsent, charcnt, wordcnt, maxwordlen, maxsenlen, kchr, kwrd, xchr, xwrd, y)= read("tweets_clean.txt")

        # # print(maxsenlen)
        # # print(maxwordlen)
        # # print(y)
        # self.maxsenlen = maxsenlen
        # self.maxwordlen = maxwordlen
        # self.wordsmap = wordsmap
        # self.charmap = charmap
        # self.charcnt = charcnt
        # self.wordcnt=wordcnt
        # self.kwrd = kwrd
        # self.kchr = kchr
        
        
        # dimword = 30
        dimword = 30
        dimchar = 5
        clword = 300
        clchar = 50
        kword = kwrd
        kchar = kchr
        
        
        
        
        datatrainword, datatestword, datatrainchar, datatestchar, targettrain, targettest = train_test_split(xwrd, xchr,y,random_state=1234, test_size=0.2)

        # print(len(datatestword))
        
        xwrd_val = []
        xchr_val = []
        y_val = []
        wrdcnt_val = wordcnt
        chrcnt_val = charcnt
        # print(wrdcnt_val)  
        # print(chrcnt_val)
        
        
        self.inpsentences = []
        self.entries =[]
        
        print("Say the number of sentences you want to enter:)")
        self.numsen_test = input()
        self.Master= tk.Tk()
        self.Master.geometry("500x200")
        
        for n in range(int(self.numsen_test)):
            putsen_str = "Sentence number" + str(n+1)
            l = tk.Label(self.Master,text = putsen_str,width= 100)
            l.pack()
            self.entries.append(tk.Entry(self.Master,width=400))
            self.entries[n].pack()
             
        
        # self.Entry2= tk.Entry(self.Master)
        # self.Entry2.pack()

        self.Button= tk.Button(self.Master,text="Submit",command=self.Return)
        self.Button.pack()            

        self.Master.mainloop()
        
        
        # print(self.inpsentences)
        # for n in range(int(self.numsen_test)):
        for sen in self.inpsentences:
            print(sen)
            # print("Enter the desired sentence.")
            testsen = sen
            words = testsen.split()
            # print(words)
            wordmat = [0] * (maxsenlen)
            # print(wordmat)
            charmat = numpy.zeros((maxsenlen, maxwordlen))
            # print(charmat)
            for i in range(len(words)):
                if words[i] in list(wordsmap.keys()):
                    # print(self.wordsmap[words[i]])
                    wordmat[int((kwrd / 2) + i)] = wordsmap[words[i]]
                else:
                    # wordcnt = wordcnt+1
                    wrdcnt_val = wrdcnt_val +1
                    wordmat[int((kwrd / 2) + i)] = -1
                # print(wordmat)
                for j in range(len(words[i])):
                    if words[i][j] in list(charmap.keys()):
                        charmat[int((kwrd / 2) + i)][int((kchr / 2) + j)] = charmap[words[i][j]]
                    else:
                        # charcnt = charcnt + 1
                        chrcnt_val = chrcnt_val +1
                        charmat[int((kwrd / 2) + i)][int((kchr / 2) + j)] =  -1
        # print(wordmat)
        # print(charmat)
        
        
            for i in range(batchsize):
                xwrd_val.append(wordmat) 
        
            for i in range(batchsize):
                xchr_val.append(charmat)
                y_val.append(0)
                
        # print(len(xwrd_val))  
        # print(len(xchr_val))
        # print(len(y_val))
        
        
        xvalword = theano.shared(np.asarray(xwrd_val, dtype='int16'), borrow=True)
        xvalchar = theano.shared(np.asarray(xchr_val, dtype='int16'), borrow=True)
        yval = theano.shared(np.asarray(y_val, dtype='int8'), borrow=True)
        
        xtrainword = theano.shared(np.asarray(datatrainword, dtype='int16'), borrow=True)
        xtrainchar = theano.shared(np.asarray(datatrainchar, dtype='int16'), borrow=True)
        ytrain = theano.shared(np.asarray(targettrain, dtype='int8'), borrow=True)
        xtestword = theano.shared(np.asarray(datatestword, dtype='int16'), borrow=True)
        xtestchar = theano.shared(np.asarray(datatestchar, dtype='int16'), borrow=True)
        ytest = theano.shared(np.asarray(targettest, dtype='int8'), borrow=True)
        
        self.ntrainbatches = xtrainword.get_value(borrow=True).shape[0] / batchsize
        self.ntestbatches = xtestword.get_value(borrow=True).shape[0] / batchsize
        
        
#         # final_t = T.iscalar('final_t')
#         # if (T.eq(final_t, 0)):
#         #     batchsize = 1
            
            
#         # print(batchsize)
        
        xwrd = T.wmatrix('xwrd')
        xchr = T.wtensor3('xchr')
        y = T.bvector('y')
        train = T.iscalar('train')
        index = T.iscalar()
        
        
#         # print("***********")
#         # print(xtestword.get_value()[-1])
#         # print("***********")

#         # print(xwrdtest[-1])
#         # print("***********")
        
        
        
        layercharembedinput = xchr
        layercharembed = EmbedIDLayer(
            rng,
            input=layercharembedinput,
            ninput=charcnt,
            noutput=dimchar
        )
        

        layer1input = layercharembed.output.reshape(
            (batchsize * maxsenlen, 1, maxwordlen, dimchar)
        )
        layer1 = ConvolutionalLayer(
            rng,
            layer1input,
            filter_shape=(clchar, 1, kchar, dimchar),
            image_shape=(batchsize * maxsenlen, 1, maxwordlen, dimchar)
        )
        

# maxpooling for xwh
        layer2 = MaxPoolingLayer(
            layer1.output,
            poolsize=(maxwordlen - kchar + 1, 1)
        )
        
        
        
        # embeddings_index = {}
        # f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
        # for line in f:
        #     values = line.split()
        #     word = values[0]
        #     coefs = np.asarray(values[1:], dtype='float32')
        #     embeddings_index[word] = coefs
        # f.close()
        # print('Found %s word vectors.' % len(embeddings_index))
        
        # embedding_matrix = np.zeros((wordcnt, dimword))
        # for word, i in wordsmap.items():
        #     embedding_vector = embeddings_index.get(word)
        #     if embedding_vector is not None:
        #         # words not found in embedding index will be all-zeros.
        #         embedding_matrix[i] = embedding_vector
        
        # embedding_matrix_s = theano.shared(np.asarray(embedding_matrix, dtype='int16'), borrow=True)

        
        
        # word level embedding
        layerwordembedinput = xwrd
        layerwordembed = EmbedIDLayer(
            rng,
            layerwordembedinput,
            ninput=wordcnt,
            noutput=dimword,
            # W = embedding_matrix_s
        )
        
        
        layer3wordinput = layerwordembed.output.reshape((batchsize, 1, maxsenlen, dimword))
        layer3charinput = layer2.output.reshape((batchsize, 1, maxsenlen, clchar))
        layer3input = T.concatenate(
            [layer3wordinput,
              layer3charinput],
            axis=3
        )
        layer3 = ConvolutionalLayer(
            rng,
            layer3input,
            filter_shape=(clword, 1, kword, dimword + clchar),
            image_shape=(batchsize, 1, maxsenlen, dimword + clchar),
            activation=activation
        )

        layer4 = MaxPoolingLayer(
            layer3.output,
            poolsize=(maxsenlen - kword + 1, 1)
        )
        
        layer5input = layer4.output.reshape((batchsize, clword))
        layer5 = FullyConnectedLayer(
            rng,
            dropout(rng, layer5input, train),
            ninput=clword,
            noutput=50,
            activation=activation
        )
        layer6input = layer5.output
        layer6 = FullyConnectedLayer(
            rng,
            dropout(rng, layer6input, train, p=0.1),
            ninput=50,
            noutput=2,
            activation=None
        )
       
        
        result = Result(layer6.output, y)
        loss = result.negativeloglikelihood()
        accuracy = result.accuracy()
        
#         # print("loss")

        params = layer6.params + layer5.params + layer3.params + layerwordembed.params + layer1.params + layercharembed.params
#         # params = layer6.params + layer5.params + layer3.params + layer1.params + layercharembed.params

        updates = RMSprop(learningrate=0.0005, params=params).updates(loss)
        
        self.trainmodel = theano.function(
            inputs=[index],
            outputs=[loss, accuracy],
            updates=updates,
            givens={
                xwrd: xtrainword[index*batchsize: (index+1)*batchsize],
                xchr: xtrainchar[index*batchsize: (index+1)*batchsize],
                y: ytrain[index*batchsize: (index+1)*batchsize],
                train: np.cast['int32'](1)
            }
        )
        
        self.testmodel = theano.function(
            inputs=[index],
            outputs=[loss, accuracy,layer6.output],
            givens={    
                xwrd: xtestword[index*batchsize: (index+1)*batchsize],
                xchr: xtestchar[index*batchsize: (index+1)*batchsize],
                y: ytest[index*batchsize: (index+1)*batchsize],
                train: np.cast['int32'](0)
                # final_t: np.cast['int32'](0)
            }
        )
        
        self.valmodel = theano.function(
            inputs=[index],
            outputs=[loss, accuracy,layer6.output],
            givens={    
                xwrd: xvalword[index*batchsize: (index+1)*batchsize],
                xchr: xvalchar[index*batchsize: (index+1)*batchsize],
                y: yval[index*batchsize: (index+1)*batchsize],
                train: np.cast['int32'](0)
                # final_t: np.cast['int32'](0)
            }
        )
       
    def Return(self):
        print(int(self.numsen_test))
        for n in range(int(self.numsen_test)):
            # print(self.entries[n].get())
            self.inpsentences.append(self.entries[n].get())
        
        self.Master.destroy()
        print("yayyyyy")
        
    def trainandtest(self, nepoch=4):
        epoch = 0
        test_accuracies = []
        test_losses = []
        
        train_accuracies = []
        train_losses = []
        
        y_axis = []
        d=0
        y_predictions = []
        
        print(self.ntestbatches)
        print(self.ntrainbatches)
        while epoch < nepoch:
            epoch += 1
            sumloss = 0
            sumaccuracy = 0
            for batchindex1 in range(int(self.ntrainbatches)):
                print('train batch = {}'.format(batchindex1))
                batchloss_t, batchaccuracy_t = self.trainmodel(batchindex1)
                train_accuracies.append(batchaccuracy_t)
                train_losses.append(batchloss_t)
                sumloss = 0
                sumaccuracy = 0
                # y_predictions.clear()
                for batchindex2 in range(int(self.ntestbatches)):
                    # print('test batch = {}'.format(batchindex2))
                    batchloss, batchaccuracy,c = self.testmodel(batchindex2)
                    sumloss += batchloss
                    sumaccuracy += batchaccuracy
                    # y_predictions.append(T.nnet.softmax(c).eval()[-1])
                    # print("************")
                    # print(T.nnet.softmax(c).eval()[-1])
                    # print("************")
                    
                
                y_axis.append(d)
                d = d+1
                loss = sumloss / self.ntestbatches
                accuracy = sumaccuracy / self.ntestbatches
                test_accuracies.append(accuracy)
                test_losses.append(loss)

                print('epoch: {}, test mean loss={}, test mean accuracy={},train batch loss={},train batch accuracy={}'.format(epoch, loss, accuracy,batchloss_t, batchaccuracy_t))
                print('')
            
        for i in range(int(self.numsen_test)):
            batchloss, batchaccuracy,c = self.valmodel(i)
            y_predictions.append(T.nnet.softmax(c).eval()[-1])
            print(T.nnet.softmax(c).eval()[-1])
            # print(T.nnet.softmax(c).eval()[-1][0])
            # print(T.nnet.softmax(c).eval()[-1][1])
        
        
        self.Master= tk.Tk()
        self.Master.geometry("500x200")
        for i in range(int(self.numsen_test)):
            s = "sentence number " + str(i+1) +": "+self.inpsentences[i]
            l0 = tk.Label(self.Master,text = s)
            l0.pack()
        l0 = tk.Label(self.Master,text = "")
        l0.pack()
        for i in range(int(self.numsen_test)):
            s = "The result of sentimental analysis of sentence number " + str(i+1)
            l1 = tk.Label(self.Master,text = s)
            l1.pack()
            # l0 = tk.Label(self.Master,text = self.inpsentences[i])
            # l0.pack()
            if  y_predictions[i][0] > y_predictions[i][1]:
                s = "negative"
            elif y_predictions[i][0] < y_predictions[i][1]:
                s = "positive"
                
            elif y_predictions[i][0] == y_predictions[i][1]:
                s = "natural"
                
            l = tk.Label(self.Master,text = s)
            l.pack()
        # for n in range(int(self.numsen_test)):
        #     self.entries.append(tk.Entry(self.Master))
        #     self.entries[n].pack()
        
        # # self.Entry2= tk.Entry(self.Master)
        # # self.Entry2.pack()

        # self.Button= tk.Button(self.Master,text="Submit",command=self.Return)
        # self.Button.pack()            

        self.Master.mainloop()
        
            
            
        # y_predictions.reverse()
        # for i in range(int(self.numsen_test)):
        #     print(y_predictions[int(self.numsen_test)-i-1])
        #     # del y_predictions[int(self.numsen_test)-i]
        
        return test_accuracies , test_losses ,train_accuracies , train_losses , y_axis,self.ntrainbatches
if __name__ == '__main__':
    random_state = 1234
    rng = np.random.RandomState(random_state)
    charscnn = CharSCNN(rng, batchsize=100,activation=relu)
    test_accuracies,test_losses,train_accuracies , train_losses,y_axis,ntrainbatches = charscnn.trainandtest(nepoch=1)
    # acuracies,losses,y_axis = charscnn.trainandtest(nepoch=1)
    
    
    # x_scales = []
    # for i in range(int(len(y_axis)/ntrainbatches)+1):
    #     x_scales.append(i)
        
    plt.plot(y_axis,test_accuracies, label='test accuracy')
    plt.plot(y_axis,train_accuracies,label='train accuracy')
    
    # plt.xlim([0,int(len(y_axis)/ntrainbatches)*len(y_axis)])
    plt.ylim([0,1])
    # plt.xticks(x_scales)

    ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    plt.xlabel('time') 
    plt.ylabel('accuracy') 
    plt.title('test/train accuracy') 
    plt.legend()
    plt.show() 
    
    # plt2.plot(y_axis,train_accuracies,color ="blue")
    # # plt.xlim([0,1000])
    # plt2.ylim([0,2])

    # ax2 = plt2.gca()
    # ax2.axes.xaxis.set_visible(False)
    # plt2.xlabel('time') 
    # plt2.ylabel('accuracy') 
    # plt2.title('train accuracy') 
    # plt2.show() 
    
    
    plt3.plot(test_losses,label='test loss')
    plt3.plot(train_losses,label='train loss')
    # plt2.xlim([0,1000])
    plt3.ylim([0,1])
    # plt3.xticks(x_scales)
    ax3 = plt3.gca()
    # ax3.axes.xaxis.set_visible(False)
    plt3.xlabel('time') 
    # naming the y axis 
    plt3.ylabel('loss') 
    # giving a title to my graph 
    plt3.title('test/train loss') 
    plt3.legend()
    # function to show the plot 
    plt3.show() 
    
    # plt4.plot(y_axis,train_losses,color ="pink")
    # # plt.xlim([0,1000])
    # plt4.ylim([0,2])

    # ax4 = plt4.gca()
    # ax4.axes.xaxis.set_visible(False)
    # plt4.xlabel('time') 
    # plt4.ylabel('accuracy') 
    # plt4.title('train loss') 
    # plt4.show() 

        