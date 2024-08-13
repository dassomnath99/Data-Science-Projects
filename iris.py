#Importing files
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

# Step 1: Data Preprocessing.
f_path="G:\Internships\Cypher Technologies\Project - 1\Iris Flower - Iris.csv"
iris_df=pd.read_csv(f_path)

iris_df.drop('Id',axis=1,inplace=True)#this will remove id column and modify the original dataframe

A=iris_df.drop('Species',axis=1)# it will remove the species and only the 4 features are available but it will not modify the original dataframe
B=iris_df['Species'] # it will only place the species column into B

# Step 2 : Encoding the Species Label
label_encoder=LabelEncoder() # Label Encoder changes the categorial value into numberic value

b_encoded=label_encoder.fit_transform(B)#label_encoder fit and tranform the values of B into the numeric value and place it to b_encoded

#Step 3 : Spliting the dataset into train and test

a_train,a_test,b_train,b_test=train_test_split(A,B,test_size=0.33,random_state=42)
# it will split the data into train and test set

#Step 4: Model Training

irismodel=RandomForestClassifier(random_state=42) #it will use the RandomForestClassifer into irismodel

irismodel.fit(a_train,b_train)# we are fiting the model with a_train and b_train set

#Step 5: Model Evaluation

b_pred=irismodel.predict(a_test)#now it will predict the irismodel with a_test dataset

accuracy=accuracy_score(b_pred,b_test) #now it will compare the predicted labels with true labels
print(f'Accuracy: {accuracy}')

classifi_rep=classification_report(b_test,b_pred, target_names=label_encoder.classes_)#here we use a classification report
print(classifi_rep)






