'''
Deciding whether the breast cancer is Benign or Malignant with the given information
Data - http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
Logistic Regression Neural Network with 4 layers
Layer 1 - 8 notdes
Layer 2 - 5 nodes
Layer 3 - 7 nodes
Layer 4 - 1 node
Author : Vedant Tilwani
'''


import numpy as np
import matplotlib
import tensorflow as tf
import csv

#CREATE TRAIN(80%) AND TEST(20%) SETS
def splitdata(data):
	len_o = len(data['ID'])
	len_tr = int(np.floor(0.8*len_o))
	len_te = len_o - len_tr
	print(len_o,',',len_tr,',',len_te)
	listx = (data['ID'])[:len_tr]
	a = [(data['ID'])[:len_tr]]
	a.append((data['ClumpThickness'])[:len_tr])
	a.append((data['UniformityofCellSize'])[:len_tr])
	a.append((data['UniformityofCellShape'])[:len_tr])
	a.append((data['MarginalAdhesion'])[:len_tr])
	a.append((data['SingleEpithelialCellSize'])[:len_tr])
	a.append((data['BareNuclei'])[:len_tr])
	a.append((data['BlandChromatin'])[:len_tr])
	a.append((data['NormalNucleoli'])[:len_tr])
	a.append((data['Mitoses'])[:len_tr])
	b = [(data['Class'])[:len_tr]]
	c = [(data['ID'])[:len_te]]
	c.append((data['ClumpThickness'])[:len_te])
	c.append((data['UniformityofCellSize'])[:len_te])
	c.append((data['UniformityofCellShape'])[:len_te])
	c.append((data['MarginalAdhesion'])[:len_te])
	c.append((data['SingleEpithelialCellSize'])[:len_te])
	c.append((data['BareNuclei'])[:len_te])
	c.append((data['BlandChromatin'])[:len_te])
	c.append((data['NormalNucleoli'])[:len_te])
	c.append((data['Mitoses'])[:len_te])
	d = [(data['Class'])[:len_te]]
	return a,b,c,d

def nnmodel(train_x,train_y,test_x,test_y,learning_rate=0.03):
	#Input
	m = len(train_x[0])
	X = tf.placeholder(tf.float32,[len(train_x),None],name="X")
	Y = tf.placeholder(tf.float32,[len(train_y),None],name="Y")
	#Set weights
	W1 = tf.get_variable("W1",[8,len(train_x)],initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable("b1",[8,1],initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2",[5,8],initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2",[5,1],initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3",[7,5],initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable("b3",[7,1],initializer = tf.zeros_initializer())
	W4 = tf.get_variable("W4",[1,7],initializer = tf.contrib.layers.xavier_initializer())
	b4 = tf.get_variable("b4",[1,1],initializer = tf.zeros_initializer())
	print('------WEIGHTS SET------')
	#Calculate Z and A
	Z1 = tf.add(tf.matmul(W1, X), b1) 
	print(np.shape(W1),np.shape(X),';;;;;;;;')
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2,A1),b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3,A2),b3)
	A3 = tf.nn.relu(Z3)
	Z4 = tf.add(tf.matmul(W4,A3),b4)
	A4 = tf.nn.sigmoid(Z4)
	print('------Z AND A CALCULATED------')
	#Cost function
	logits = tf.transpose(A4)
	labels = tf.transpose(Y)
	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
	print('cost = ', cost,'===========')
	print('------COST FUNCTION COMPLETED------')
	#Building the model and backward propagation
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.03).minimize(cost)
	print('------GRADIENT DESCENT OPTIMIZER COMPLETED------')
	#Initialize all the variables	
	print('------TENSORFLOW VARIABLES INITIALIZED------')
	#Create session
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		print(sess.run([cost,optimizer],feed_dict={X:train_x,Y:train_y}))
		print('------Model has been trained------')
		correct_prediction = tf.equal(tf.argmax(Z4),tf.argmax(Y))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
		print("Train Accuracy:", accuracy.eval({X: train_x, Y: train_y}))
		print("Test Accuracy:", accuracy.eval({X: test_x, Y: test_y}))

	return 0

#EXTRACTING DATA AND STROING IT IN A DICTONARY
with open('breast-cancer-wisconsin.data','rt') as f:
	reader = csv.reader(f,delimiter=',')
	lineData = list()

	cols = next(reader)

	for col in cols:
		lineData.append(list())


	for line in reader:
		for i in range(0, len(lineData)):
      		# Copy the data from the line into the correct columns.
			#print(line[i])
			lineData[i].append(line[i])

	data1 = dict()

	for i in range(0, len(cols)):
    	# Create each key in the dict with the data in its column.
		data1[cols[i]] = lineData[i]

data = {'ID': data1['1000025'],
		'ClumpThickness': data1['5'],
		'UniformityofCellSize': data1['1'],
		'UniformityofCellShape': data1['1'],
		'MarginalAdhesion': data1['1'],
		'SingleEpithelialCellSize': data1['2'],
		'BareNuclei': data1['1'],
		'BlandChromatin': data1['3'],
		'NormalNucleoli': data1['1'],
		'Mitoses': data1['1'],
		'Class':data1['2']}

train_x, train_y, test_x, test_y = splitdata(data)

print(len(train_x[0]))


for i in range(len(train_y[0])):
	if (train_y[0][i] == '2'):
		train_y[0][i] =0
	else:
		train_y[0][i] =1
for i in range(len(test_y[0])):
	if (test_y[0][i] == '2'):
		test_y[0][i] =0
	else:
		test_y[0][i] =1

final_train = train_x[1:len(train_x)]
train_x = final_train

final_test = test_x[1:len(test_x)]
test_x = final_test

print(np.shape(train_x))

y = nnmodel(train_x,train_y,test_x,test_y)