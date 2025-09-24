# Binary Classification with Neural Networks on the Census Income Dataset

## Algorithm: 

### Step 1: Import Required Libraries

Import libraries for:

Data Handling: pandas, numpy

Data Preprocessing: sklearn.preprocessing, sklearn.model_selection

PyTorch Core: torch, torch.nn, torch.optim

Utilities: random, os (for reproducibility)

### Step 2: Load and Inspect the Dataset

Load income.csv using pandas.

Check for:

Missing values (df.isnull().sum())

Data types of each column (df.dtypes)

Unique values in categorical columns (df[col].unique()).

### Step 3: Data Preprocessing

Identify:

Categorical columns (e.g., workclass, education, marital-status…)

Continuous columns (e.g., age, hours-per-week, capital-gain…)

Target label (income).

Encode Categorical Variables

For each categorical column:

Create a mapping dictionary {category: index}.

Replace category values with their corresponding index.

Normalize Continuous Variables

Compute mean and standard deviation for each continuous column.

Standardize using:

𝑥𝑛𝑜𝑟𝑚=𝑥−mean/std
		​

Prepare Target Labels

Convert income:

<=50K → 0

>50K → 1.

### Step 4: Train/Test Split

Split the dataset into:

Training set: 25,000 samples

Testing set: 5,000 samples
using train_test_split() with a fixed random seed.

Convert all splits to NumPy arrays.

### Step 5: Convert Data to PyTorch Tensors

Convert:

Categorical arrays → torch.LongTensor

Continuous arrays → torch.FloatTensor

Labels → torch.LongTensor

### Step 6: Define the Tabular Model

Create a TabularModel class (inherits from torch.nn.Module).

Components:

Embedding Layers

For each categorical feature, create an embedding:

Embedding(num_categories,embedding_dim)
Embedding(num_categories,embedding_dim)

Batch Normalization for continuous features:

nn.BatchNorm1d(num_continuous_features)

Fully Connected Layers:

Input size = (sum of embedding dims + number of continuous features)

Hidden layer with 50 neurons:

Linear layer → nn.Linear(input_dim, 50)

Activation → nn.ReLU()

Dropout → nn.Dropout(p=0.4)

Output layer:

Linear layer → nn.Linear(50, 2) (binary classification with 2 classes)

Forward Pass:

Pass each categorical column through its embedding.

Concatenate embeddings and continuous features.

Apply batch normalization.

Pass through hidden and output layers.

Return raw logits.

### Step 7: Initialize Model, Loss Function & Optimizer

Set a random seed (torch.manual_seed()).

Create a TabularModel instance.

Define:

Loss function: nn.CrossEntropyLoss()

Optimizer: torch.optim.Adam(model.parameters(), lr=0.001)

### Step 8: Training Loop

Repeat for 300 epochs:

Set model to training mode: model.train().

Zero the gradients: optimizer.zero_grad().

Forward pass:

outputs = model(categorical_train, continuous_train)

Compute loss:

loss = criterion(outputs, labels_train)

Backpropagation:

loss.backward()

Update weights:

optimizer.step()

Optionally print training loss every few epochs for monitoring.

Step 9: Evaluation on Test Set

Set model to evaluation mode: model.eval().

Disable gradient calculation with torch.no_grad().

Forward pass on the test set:

outputs = model(categorical_test, continuous_test)

Compute:

Test Loss: criterion(outputs, labels_test)

Predictions: torch.argmax(outputs, dim=1)

Accuracy:

accuracy=(predictions == labels).sum()/total samples	​

### Step 10 : User Input Prediction Function

Create a function predict_income(user_input_dict):

Accepts new data as a dictionary of feature names and values.

Encodes categorical inputs and normalizes continuous inputs using training statistics.

Converts inputs to tensors.

Runs a forward pass through the trained model.

Returns:

1 → Earning > $50K

0 → Earning ≤ $50K
