model1=models.Sequential()
model1.add(Conv2D(32, (5, 5), input_shape = (32, 32, 1)))
model1.add(LeakyReLU(alpha=0.1))
model1.add(MaxPooling2D(pool_size = (2, 2)))
model1.add(Conv2D(128, (5, 5)))
model1.add(LeakyReLU(alpha=0.1))
model1.add(Conv2D(64, (5, 5)))
model1.add(LeakyReLU(alpha=0.1))
model1.add(Conv2D(32, (5, 5)))
model1.add(LeakyReLU(alpha=0.1))
model1.add(MaxPooling2D(pool_size = (2, 2)))
model1.add(Flatten())
model1.add(Dense(1000))
model1.add(LeakyReLU(alpha=0.1))
model1.add(Dropout(0.5))
model1.add(Dense(500))
model1.add(LeakyReLU(alpha=0.1))
model1.add(Dropout(0.5))
model1.add(Dense(250))
model1.add(LeakyReLU(alpha=0.1))
model1.add(Dense(1, activation = 'sigmoid'))
model1.summary()

model1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history1 = model1.fit(X_train1, y_train1, epochs= 100, 
                    validation_split = 0.1)

print("Loss of the model is - " , model1.evaluate(X_test1,y_test1)[0])
print("Accuracy of the model is - " , model1.evaluate(X_test1,y_test1)[1]*100 , "%")

"""plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')"""
epochs = [i for i in range(100)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()
test_loss, test_acc = model.evaluate(X_test1,  y_test1, verbose=2)

predictions = model.predict_classes(X_test1)
predictions = predictions.reshape(1,-1)[0]

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test1, predictions, target_names = ['Tumor(Class 0)','Normal (Class 1)']))

cm = confusion_matrix(y_test1,predictions)
cm

correct = np.nonzero(predictions == y_test1)[0]
incorrect = np.nonzero(predictions != y_test1)[0]

i = 0
for c in correct[:6]:
    plt.subplot(3,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[c].reshape(150,150), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y_test[c]))
    plt.tight_layout()
    i += 1

i = 0
for c in incorrect[:6]:
    plt.subplot(3,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[c].reshape(150,150), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y_test[c]))
    plt.tight_layout()
    i += 1

img=cv2.imread('/content/drive/MyDrive/brain_tumor_dataset/Y1.jpg', cv2.IMREAD_GRAYSCALE)
img1 = getPixels(img, (300,300))
img = getPixels(img, (32, 32))
img1 = img1/255.0
img = img / 255.0
fig , ax = plt.subplots(1,2)
#ax[0].figure()
ax[0].set_title('32x32 Image')
ax[0].imshow(img)
#ax[1].figure()
ax[0].set_title('150x150 Image')
ax[1].imshow(img1)

img = img.reshape(-1, 32, 32, 1)
p = model.predict(img)
print('The predicted class is ',p[0][0])

import matlab.engine
eng = matlab.engine.start_matlab()

p = model.predict(img)
if p[0][0] == 1.0:
    eng.tumor(img1)
