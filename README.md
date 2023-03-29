# Mobile price classification with multilayer perceptron

## Authors
 - [Erik Matovič](https://github.com/Matovic)
 - Jakub Horvat 

## Usage

[Install tensorflow to enable GPU](https://www.tensorflow.org/install/pip)   

## Solution
### 1. Exploratory Data Analysis

[Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?select=train.csv)

Attributes:
 - battery_power - Total energy a battery can store in one time measured in mAh. Range: 501-1998, int64
 - blue - Has bluetooth or not. Range: 0-1, int64
 - clock_speed - speed at which microprocessor executes instructions. Range: 0.5-3, float64
 - dual_sim - Has dual sim support or not. Range: 0-1, int64
 - fc - Front Camera mega pixels. Range: 0-19, int64
 - four_g - Has 4G or not. Range: 0-1, int64
 - int_memory - Internal Memory in Gigabytes. Range: 2-64, int64
 - m_dep - Mobile Depth in cm. Range: 0.1-1, float64
 - mobile_wt - Weight of mobile phone. Range: 80-200, int64
 - n_cores - Number of cores of processor. Range: 1-8, int64
 - pc - Primary Camera mega pixels. Range: 0-20, int64
 - px_height - Pixel Resolution Height. Range: 0-1960, int64
 - px_width - Pixel Resolution Width. Range: 500-1998, int64
 - ram - Random Access Memory in Mega Bytes. Range: 256-3998, int64
 - sc_h - Screen Height of mobile in cm. Range: 5-19, int64
 - sc_w - Screen Width of mobile in cm. Range: 0-18, int64
 - talk_time - longest time that a single battery charge will last. Range: 2-20, int64
 - three_g - Has 3G or not. Range: 0-1, int64
 - touch_screen - Has touch screen or not. Range: 0-1, int64
 - wifi - Has wifi or not. Range: 0-1, int64
 - price_range - This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost). Range: 0-3, int64

Pairplot:
 <p align="center">
	<img src="./figures/1_EDA/pairplot.png">
</p>

Heatmap:
 <p align="center">
	<img src="./figures/1_EDA/heatmap.png">
</p>

Target variable price_range based on values of ram:
 <p align="center">
	<img src="./figures/1_EDA/ram_price_range.png">
</p>

Target variable price_range based on the count values of ram:
 <p align="center">
	<img src="./figures/1_EDA/ram_count_price_range.png">
</p>

### 2. Data Preprocessing

Based on [exploratory data analysis](./src/EDA.ipynb) test set does not have target variable price_range. We split our dataset into train-dev-test. We have train and test sets, but we split test set by half to dev-test sets. We will rougly have train-dev-test 80%-10%-10%.  


### 3. Model
Best parameters from WandB:
 - batch size: 128,
 - hidden size: 256,
 - epochs: 100,
 - learning rate: 0.03648030109469176
#### PyTorch
```python3
class MLP(nn.Module):
    """ 
    Model class.
    :param Module from torch.nn
    """
    def __init__(self, n_inputs: int, n_classes: int, lr: float, hidden_size: float) -> None:
        """
        Model elements init.
        """
        super(MLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.lr = lr
        self.hidden_size = hidden_size
    
        self.dense1 = nn.Linear(n_inputs, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense5 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense6 = nn.Linear(self.hidden_size, self.n_classes)
        self.relu = nn.ReLU()
        self.droput = nn.Dropout(p=0.05)
 
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """
        Feed forward
        """
        # input to first hidden layer
        output = self.dense1(X)
        output = self.relu(output)
        output = self.droput(output)
        
        output = self.dense2(output)
        output = self.relu(output)
        output = self.droput(output)

        output = self.dense3(output)
        output = self.relu(output)
        output = self.droput(output)
        
        output = self.dense4(output)
        output = self.relu(output)
        output = self.droput(output)
        
        output = self.dense5(output)
        output = self.relu(output)
        output = self.droput(output)

        # final layer and output
        output = self.dense6(output)

        return output
```

#### TensorFlow
```python3
#Define Layers and Dropout of MLP
hidden_size = 256
model = keras.Sequential([
    keras.layers.Dense(hidden_size,input_shape=(20,),activation='relu'),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(hidden_size,activation='relu'),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(hidden_size,activation='relu'),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(hidden_size,activation='relu'),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(4,activation='sigmoid') #activation='sigmoid'
])

#best learning rate
lr = 0.03648030109469176
#definition of optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#configuration of model for training
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy','mse']
              )
```

### 4. Training & validation
#### PyTorch
```python3
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() 
    return correct


def train_mlp(n_epochs, mlp, optimizer, loss_fn, 
              train_dl, val_dl, device, batch_size):
    # init train lists for statistics
    loss_train, accuracy_train = list(), list()

    # init validation lists for statistics
    loss_validation, accuracy_validation = list(), list()

    # enumerate epochs
    for epoch in range(n_epochs):
        # init epoch train counters
        epoch_train_accuracy, epoch_train_total, epoch_train_true, epoch_train_loss = 0, 0, 0, 0

        # init epoch validation counters
        epoch_validation_accuracy, epoch_validation_total, \
            epoch_validation_true, epoch_validation_loss = 0, 0, 0, 0

        # enumerate mini batches
        for idx, (X_batch, y_batch) in enumerate(train_dl):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # clear the gradients
            optimizer.zero_grad()
            # Make prediction logits with model
            y_logits = mlp(X_batch)
            y_pred_probs = torch.softmax(y_logits, dim=1) 
            # go from logits -> prediction probabilities -> prediction labels
            y_pred = torch.argmax(y_pred_probs, dim=1) 
            
            loss = loss_fn(y_logits, y_batch)
            loss.backward()
            # update model weights
            optimizer.step()

            # update train counters
            epoch_train_loss += loss.item()
            epoch_train_true += accuracy_fn(y_batch, y_pred)
            epoch_train_total += len(y_batch)
        
        # update train accuracy & loss statistics
        epoch_train_accuracy = (epoch_train_true/epoch_train_total) * 100
        epoch_train_loss /= (len(train_dl.dataset)/train_dl.batch_size)

        # disable gradient calculation
        with torch.no_grad():
            for idx, (X_batch, y_batch) in enumerate(val_dl):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                # compute the models output
                test_logits = mlp(X_batch)
                test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
                # calculate loss
                loss = loss_fn(test_logits, y_batch)

                # update validation counters
                epoch_validation_loss += loss.item()
                epoch_validation_true += accuracy_fn(y_batch, test_pred)
                epoch_validation_total += len(y_batch)
        
        # update validation accuracy & loss statistics
        epoch_validation_accuracy = (epoch_validation_true/epoch_validation_total) * 100
        epoch_validation_loss /= (len(val_dl.dataset)/val_dl.batch_size)

        # update global epochs statistics
        loss_train.append(epoch_train_loss)
        accuracy_train.append(epoch_train_accuracy)
        loss_validation.append(epoch_validation_loss)
        accuracy_validation.append(epoch_validation_accuracy)

        if epoch == (n_epochs - 1): 
            print(
                f'Epoch {epoch}/{n_epochs}: \
                train loss {loss_train[-1]}, \
                validation loss {loss_validation[-1]}, \
                train accuracy {accuracy_train[-1]}, \
                validation accuracy {accuracy_validation[-1]}'
            )

    return loss_train, accuracy_train, loss_validation, accuracy_validation
```

<p align="center">
	<img src="./figures/validation_acc.png">
</p>

#### TensorFlow
```python3
#model training
history = model.fit(x_train,
                    y_cat_train,
                    batch_size=128,
                    epochs=100,
                    validation_data=(x_val,y_cat_val),
                    callbacks=[
                      WandbMetricsLogger(log_freq=5),
                      WandbModelCheckpoint("models")
                    ],
                    verbose=0)
```

### 5. Testing

#### PyTorch
```python3
def evaluation(mlp, test_dl):
    y_pred_all, y_test_all = list(), list()

	# evaluate on test set
    with torch.no_grad():
        for X_batch, y_batch in test_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_hat = mlp(X_batch)
            test_pred = torch.softmax(y_hat, dim=1).argmax(dim=1)
            epoch_validation_true += accuracy_fn(y_batch, test_pred)
            
            y_pred_all.extend(test_pred.cpu().numpy())
            y_test_all.extend(y_batch.cpu().numpy())
        epoch_validation_true = (epoch_validation_true / len(test_dl.dataset)) * 100

    print('acc:', epoch_validation_true)
    report = classification_report(y_test_all, y_pred_all, target_names=['0', '1', '2', '3'], digits=4)
    print(report)
```

<p align="center">
	<img src="./figures/classification_report.png">
</p>

#### TensorFlow
```python3
predicted_test = []
for row in predict:
    a = max(row)
    b=np.where(row==a)[0][0]
    predicted_test.append(b)

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
report = classification_report(y_test, predicted_test, target_names=['0', '1', '2', '3'], output_dict=True)
f = plt.figure(figsize=(10,6)) #plotting
f.set_size_inches(18.5, 10.5)
f.set_dpi(100)
print(report)
```
<p align="center">
	<img src="./figures/classification_report_tensor.png">
</p>

### 6. Classification
Classification on unknown data.  
```python3
matrix = confusion_matrix(y_test_all, y_pred_all)
matrix_display = ConfusionMatrixDisplay(matrix, display_labels=['0', '1', '2', '3'])
matrix_display.plot(cmap='Blues')
```

<p align="center">
	<img src="./figures/classification.png">
</p>

## Conclusion
We have implemented mutlilayer perceptron using PyTorch and tensorflow. Training was on the GPU using CUDA. 

