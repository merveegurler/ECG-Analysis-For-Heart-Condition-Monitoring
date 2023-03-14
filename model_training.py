import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout


counter = 0
a = 0
rows = []
cols = [1, 4, 5, 6, 7, 8, 10, 12, 18, 19, 20, 21, 22, 24, 26]

train_data = pd.read_csv('Assets/dataset/train/train.csv')

train_data.replace(['VEB', 'SVEB', 'F', 'Q'], 0, inplace=True)
train_data.replace(['N'], 1, inplace=True)

train_data = train_data.astype('float32')
train_data = train_data.to_numpy()

for i in range(0, len(train_data)):
    if train_data[i, 1] == 1.0 and counter >= 22000:
        rows.append(i+1)
    if train_data[i, 1] == 1.0:
        counter += 1

dataframe_train = pd.read_csv("Assets/dataset/train/train.csv", skiprows=rows, usecols=cols)

counter = 0
rows = []

test_data = pd.read_csv('Assets/dataset/test/test.csv')

test_data.replace(['VEB', 'SVEB', 'F', 'Q'], 0, inplace=True)
test_data.replace(['N'], 1, inplace=True)

test_data = test_data.astype('float32')
test_data = test_data.to_numpy()

for i in range(0, len(test_data)):
    if test_data[i, 1] == 1.0 and counter >= 11000:
        rows.append(i+1)
    if test_data[i, 1] == 1.0:
        counter += 1

dataframe_test = pd.read_csv("Assets/dataset/test/test.csv", skiprows=rows, usecols=cols)

dataframe_test.replace(['VEB', 'SVEB', 'F', 'Q'], 0, inplace=True)
dataframe_test.replace(['N'], 1, inplace=True)

dataframe_test = dataframe_test.astype('float32')
dataframe_test = dataframe_test.to_numpy()

dataframe_train.replace(['VEB', 'SVEB', 'F', 'Q'], 0, inplace=True)
dataframe_train.replace(['N'], 1, inplace=True)

dataframe_train = dataframe_train.astype('float32')
dataframe_train = dataframe_train.to_numpy()

print(dataframe_train.shape)
print(dataframe_test.shape)

training_features = dataframe_train[:, 1:]
training_labels = dataframe_train[:, 0]

testing_features = dataframe_test[:, 1:]
testing_labels = dataframe_test[:, 0]

print(training_features.shape)
print(training_features[0].shape)
print(training_labels.shape)

print(testing_features.shape)
print(testing_features[0].shape)
print(testing_labels.shape)

mean = training_features.mean(axis=0)
training_features -= mean
std = training_features.std(axis=0)
training_features /= std

mean = testing_features.mean(axis=0)
testing_features -= mean
std = testing_features.std(axis=0)
testing_features /= std

model = Sequential()
model.add(Input(shape=(14,)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=training_features,
          y=training_labels,
          epochs=256,
          validation_data=(testing_features, testing_labels))

model.save('Assets/Model/cnn_model.h5')

results = model.evaluate(testing_features, testing_labels)

print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))
