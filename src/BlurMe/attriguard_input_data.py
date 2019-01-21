import numpy as np
from random import shuffle

train_app_filepath="./data/train_user.txt"
train_label_filepath="./data/train_label.txt"
test_app_filepath="./data/test_user.txt"
test_label_filepath="./data/test_label.txt"

class Data_Class:
    def __init__(self):
        self.number_of_train=14607
        self.number_of_app=10000
        self.number_of_label=25
        self.number_of_test=1631
#####################################################
    def input_train_app(self):
        self.train_app_ids = []
        self.train_app=np.zeros([self.number_of_train,self.number_of_app],dtype=np.float32)
        input_train_file=open(train_app_filepath,'r')
        i=0
        for line in input_train_file:
            line=line[0:-1]
            line_app_ids = []

            for appnum in line.split(' '):
                appnumdetail=appnum.split(':')
                self.train_app[i,int(appnumdetail[0])]=float(appnumdetail[1])
                line_app_ids.append(int(appnumdetail[0]))

            shuffle(line_app_ids)
            self.train_app_ids.append(line_app_ids)
            i+=1
            
        input_train_file.close()
#####################################################
    def input_test_app(self):
        self.test_app_ids = []
        self.test_app=np.zeros([self.number_of_test,self.number_of_app],dtype=np.float32)
        input_test_file=open(test_app_filepath,'r')
        i=0
        for line in input_test_file:
            line=line[0:-1]
            line_app_ids = []
            for appnum in line.split(' '):
                appnumdetail=appnum.split(':')
                self.test_app[i,int(appnumdetail[0])]=float(appnumdetail[1])
                line_app_ids.append(int(appnumdetail[0]))

            shuffle(line_app_ids)
            self.test_app_ids.append(line_app_ids)
            i+=1

        input_test_file.close()
        
#####################################################
    def old_input_train_label(self):
        self.train_label=np.full([self.number_of_train,self.number_of_label],0,dtype=np.float32)
        input_train_label=np.loadtxt(train_label_filepath,dtype=np.int)
        for i in np.arange(self.train_label.shape[0]):
            self.train_label[i,input_train_label[i]]=1

#####################################################
    def old_input_test_label(self):
        self.test_label=np.full([self.number_of_test,self.number_of_label],0,dtype=np.float32)
        input_test_label=np.loadtxt(test_label_filepath,dtype=np.int)
        for i in np.arange(self.test_label.shape[0]):
            self.test_label[i,input_test_label[i]]=1

#####################################################
    def input_train_label(self):
        self.train_label=np.full([self.number_of_train],0,dtype=np.float32)
        input_train_label=np.loadtxt(train_label_filepath,dtype=np.int)
        for i in np.arange(self.train_label.shape[0]):
            self.train_label[i]=input_train_label[i]

    def input_test_label(self):
        self.test_label=np.full([self.number_of_test],0,dtype=np.float32)
        input_test_label=np.loadtxt(test_label_filepath,dtype=np.int)
        for i in np.arange(self.test_label.shape[0]):
            self.test_label[i]=input_test_label[i]

if __name__ == "__main__":
    X = Data_Class()
    X.input_train_app() # 90% users (app ratings)
    X.input_test_app() # 10% users (app ratings)
    X.input_train_label() # 90% users (city they live in)
    X.input_test_label() # 10% users (city they live in)

    print('test_app_ids', X.test_app_ids[0])
    print('test_label', X.test_label[0])

    total = 0
    i = 0
    j = 0
    k = 0
    app_ids = np.concatenate((X.train_app_ids, X.test_app_ids), axis=0)

    for x in app_ids:
        total += 1
        if len(x) > 35:
            i += 1
        if len(x) > 40:
            j += 1
        if len(x) > 45:
            k += 1

    
    print("Number of users with more than 35 ratings:", i)
    print("Number of users with more than 40 ratings:", j)
    print("Number of users with more than 45 ratings:", k)
    print("Total Number of users:", total)
    print("Total Number of apps:", X.test_app.shape[0])
    print("Total Number of cities:", np.amax(X.test_label)+1)
    print("test label: ", X.test_label[0])