
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from csv import reader
from sklearn import preprocessing
from sklearn.metrics import f1_score,accuracy_score
import time
from sklearn import linear_model

# define baseline model
def nn_model(dim1):
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=dim1, activation='relu'))
	for i in range(9):
		model.add(Dense(100, activation='relu'))
	model.add(Dense(8, activation='softmax'))
    # Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def read_embeddings(f):
    with open(f, 'r') as read_obj:
        csv_reader = reader(read_obj)
        list_of_rows = list(csv_reader)
    return list_of_rows

def get_labels(f):
    raw_data = pd.read_json(f,
                        lines=True,
                        orient='columns')
    y_label = []
    for i in range(len(raw_data)):
        if raw_data['seniority_level'][i]:
            y_label.append(raw_data['seniority_level'][i])
    y_label = np.array(y_label)
    return y_label

def evaluate_model(model,test_x,y_testCat):
    y_predCat = model.predict(test_x)
    y_predCat = np.argmax(y_predCat, axis=1)
    y_testCat = np.argmax(y_testCat, axis=1)

    accuracy = accuracy_score(y_testCat, y_predCat)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    f1Score = f1_score(y_testCat, y_predCat,average='macro')
    print('f1Score: %f' % f1Score)

train_x = read_embeddings('train_embeddings.csv')
test_x = read_embeddings('test_embeddings.csv')
print('Train_x size',len(train_x))
print('Train_x embedding size',len(train_x[0]))
print('Test_x size',len(test_x))
print('Test_x embedding size',len(test_x[0]))

y_train=get_labels("seniority.train")
y_test = get_labels("seniority.test")

print('y_train size',len(y_train))
print('y_test size',len(y_test))


le=preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_trainCat = to_categorical(y_train)
y_testCat = to_categorical(y_test)



train_x = np.array(train_x,dtype='float64')
test_x = np.array(test_x,dtype='float64')

print(train_x.shape)

## Linear regression model

regr = linear_model.LinearRegression()
start = time.time()
regr.fit(train_x, y_trainCat)
end = time.time()
print('Evaluating Linear regression model')
evaluate_model(regr,test_x,y_testCat)
print('Time taken for Linear regression', end-start)


##NN model
model = nn_model(len(train_x[0]))
start = time.time()
model.fit(train_x, y_trainCat, epochs=75)
end = time.time()
print('Evaluating Neural network model')
evaluate_model(model,test_x,y_testCat)
print('Time taken for Neural network', end-start)
