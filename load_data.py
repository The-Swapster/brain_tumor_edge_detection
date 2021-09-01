#Store the path of dataset's folder on your drive
#path store all the image paths, #labels - output of all images (Yes or No)
folder = "/content/drive/MyDrive/brain_tumor_dataset"

paths = []
labels = []
cat = []
for filename in os.listdir(folder):
  #The image names aren't uniform in the dataset, hence find function has been used
  if filename.find('n')!= -1 or filename.find('N')!= -1 :
    labels.append(0)
    cat.append('No')
  elif filename.find('Y')!= -1 or filename.find('y')!= -1 :
    labels.append(1)
    cat.append('Yes')
  
  #join complete image path
  img = os.path.join(folder,filename)
  if img is not None:
    paths.append(img)

#labels
len(labels)

sns.countplot(labels)

np.unique(labels, return_counts=True)

s = cv2.imread(paths[200])
print(s.shape)
s1 = cv2.imread(paths[5])
print(s1.shape)
#to check rough size of images to decide resizing size

def getPixels(image, size):
  pixels = cv2.resize(image,size)
  return pixels

#iterating in training set of data
rawImages1 = []
for i in paths:
    img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    #stores the raw pixel values of this image after resizing
    pixels = getPixels(img,(32,32))
    #stores the raw pixel values of images
    rawImages1.append(pixels)
    
#iterating in training set of data
rawImages = []
for i in paths:
    img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    #stores the raw pixel values of this image after resizing
    pixels = getPixels(img,(150,150))
    #stores the raw pixel values of images
    rawImages.append(pixels)
    
#has raw pixel values
X_train,X_test,y_train,y_test = train_test_split(rawImages,labels,test_size = 0.2, random_state=4)
X_train1,X_test1,y_train1,y_test1 = train_test_split(rawImages1,labels,test_size = 0.2, random_state=4)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_train1 = np.array(X_train1)
X_test1 = np.array(X_test1)
y_train1 = np.array(y_train1)
y_test1 = np.array(y_test1)

X_train.shape

#Just printing out some training images with their labels 
plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    if y_train[i] == 0: 
      c='No'
    else:
      c='Yes'
    plt.xlabel(c, fontsize=16)
plt.show()

#Just printing out some training images with their labels 
plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train1[i])
    if y_train1[i] == 0: 
      c='No'
    else:
      c='Yes'
    plt.xlabel(c, fontsize=16)
plt.show()

X_train = X_train.reshape(-1, 150, 150, 1)
X_train1 = X_train1.reshape(-1, 32, 32, 1)
X_test = X_test.reshape(-1, 150, 150, 1)
X_test1 = X_test1.reshape(-1, 32, 32, 1)

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train1 = X_train1 / 255.0
X_test1 = X_test1 / 255.0

X_train.shape
y_train.shape
