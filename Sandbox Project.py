# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
df = pd.read_csv("heart.csv")

# 1. Preview the first few records
print(df.head())

# 2. Dataset info (data types and missing values)
df.info()

# 3. Statistical summary (mean, std, min, 25%, 50%, 75%, max for numeric columns)
print(df.describe())

# 4. Distribution of the target variable
print(df.target.value_counts())

# 5. Plot distribution of the target (Heart disease)
sns.countplot(x="target", data=df, palette="bwr")
plt.show()

# 6. Gender distribution
sns.countplot(x='sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1 = male)")
plt.show()

# 7. Gender vs Heart Disease
pd.crosstab(df.sex, df.target).plot(kind="bar", figsize=(15,6), color=['#1CA53B','#AA1111'])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["No Disease", "Has Disease"])
plt.ylabel('Frequency')
plt.show()

# 8. Cholesterol vs Heart Disease
sns.boxplot(x='target', y='chol', data=df)
plt.title('Cholesterol Levels for Heart Disease vs. No Heart Disease')
plt.show()

# 9. Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.show()

# 10. Data Preprocessing (Features and target split)
X = df.drop('target', axis=1)
y = df['target']

# 11. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 12. Model Training using Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 13. Model Evaluation (Accuracy)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 14. Make predictions
y_pred = model.predict(X_test)

# 15. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 16. Precision, Recall, F1-Score
print(classification_report(y_test, y_pred))

# 17. Save the trained model
joblib.dump(model, 'heart_disease_model.pkl')

# 18. Optional: Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Model Accuracy: {rf_accuracy * 100:.2f}%")

# Optional: Tkinter GUI for prediction
def predict():
    # Get user inputs from Tkinter
    age = int(entry_age.get())
    sex = int(entry_sex.get())
    cp = int(entry_cp.get())
    trestbps = int(entry_trestbps.get())
    chol = int(entry_chol.get())
    fbs = int(entry_fbs.get())
    restecg = int(entry_restecg.get())
    thalach = int(entry_thalach.get())
    exang = int(entry_exang.get())
    oldpeak = float(entry_oldpeak.get())

    # Create an input array for prediction
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]])

    # Predict with the model
    prediction = model.predict(user_input)

    if prediction == 1:
        messagebox.showinfo("Prediction", "The person is at risk of heart disease.")
    else:
        messagebox.showinfo("Prediction", "The person is not at risk of heart disease.")

# Create GUI window
window = tk.Tk()
window.title("Heart Disease Prediction")

# Labels and entry widgets for user inputs
tk.Label(window, text="Age").pack()
entry_age = tk.Entry(window)
entry_age.pack()

tk.Label(window, text="Sex (0=female, 1=male)").pack()
entry_sex = tk.Entry(window)
entry_sex.pack()

tk.Label(window, text="Chest Pain Type (cp)").pack()
entry_cp = tk.Entry(window)
entry_cp.pack()

tk.Label(window, text="Resting Blood Pressure (trestbps)").pack()
entry_trestbps = tk.Entry(window)
entry_trestbps.pack()

tk.Label(window, text="Serum Cholesterol (chol)").pack()
entry_chol = tk.Entry(window)
entry_chol.pack()

tk.Label(window, text="Fasting Blood Sugar > 120 mg/dl (fbs)").pack()
entry_fbs = tk.Entry(window)
entry_fbs.pack()

tk.Label(window, text="Resting Electrocardiographic Results (restecg)").pack()
entry_restecg = tk.Entry(window)
entry_restecg.pack()

tk.Label(window, text="Maximum Heart Rate Achieved (thalach)").pack()
entry_thalach = tk.Entry(window)
entry_thalach.pack()

tk.Label(window, text="Exercise-Induced Angina (exang)").pack()
entry_exang = tk.Entry(window)
entry_exang.pack()

tk.Label(window, text="ST Depression (oldpeak)").pack()
entry_oldpeak = tk.Entry(window)
entry_oldpeak.pack()

# Button to trigger prediction
button = tk.Button(window, text="Predict", command=predict)
button.pack()

# Start the Tkinter event loop
window.mainloop()

# Load the dataset
df = pd.read_csv("heart.csv")

# 1. Scatter Plot: Age vs Maximum Heart Rate (Colored by Target)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='thalach', hue='target', palette='coolwarm', alpha=0.7)
plt.title('Age vs Maximum Heart Rate (Colored by Heart Disease Status)')
plt.xlabel('Age')
plt.ylabel('Maximum Heart Rate Achieved (thalach)')
plt.legend(title='Heart Disease (Target)', loc='upper right', labels=['No Disease', 'Has Disease'])
plt.show()

# 2. Exercise-Induced Angina (exang) vs Target
pd.crosstab(df.exang, df.target).plot(kind="bar", figsize=(10, 6), color=['#1CA53B','#AA1111'])
plt.title('Heart Disease Frequency for Exercise-Induced Angina')
plt.xlabel('Exercise-Induced Angina (0 = No, 1 = Yes)')
plt.xticks(rotation=0)
plt.legend(["No Disease", "Has Disease"])
plt.ylabel('Frequency')
plt.show()

# 3. Fasting Blood Sugar (fbs) vs Target
pd.crosstab(df.fbs, df.target).plot(kind="bar", figsize=(10, 6), color=['#1CA53B','#AA1111'])
plt.title('Heart Disease Frequency for Fasting Blood Sugar')
plt.xlabel('Fasting Blood Sugar (0 = No, 1 = Yes)')
plt.xticks(rotation=0)
plt.legend(["No Disease", "Has Disease"])
plt.ylabel('Frequency')
plt.show()

# 4. Chest Pain Type (cp) vs Target
pd.crosstab(df.cp, df.target).plot(kind="bar", figsize=(12, 6), color=['#1CA53B','#AA1111'])
plt.title('Heart Disease Frequency for Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation=0)
plt.legend(["No Disease", "Has Disease"])
plt.ylabel('Frequency')
plt.show()

# 5. Correlation Matrix between all features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Heart Disease Dataset Features')
plt.show()

# Load the dataset
df = pd.read_csv("heart.csv")

# 1. Scatter Plot: Age vs Maximum Heart Rate (Colored by Target)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='thalach', hue='target', palette='coolwarm', alpha=0.7)
plt.title('Age vs Maximum Heart Rate (Colored by Heart Disease Status)')
plt.xlabel('Age')
plt.ylabel('Maximum Heart Rate Achieved (thalach)')
plt.legend(title='Heart Disease (Target)', loc='upper right', labels=['No Disease', 'Has Disease'])
plt.show()

# 2. Exercise-Induced Angina (exang) vs Target
pd.crosstab(df.exang, df.target).plot(kind="bar", figsize=(10, 6), color=['#1CA53B','#AA1111'])
plt.title('Heart Disease Frequency for Exercise-Induced Angina')
plt.xlabel('Exercise-Induced Angina (0 = No, 1 = Yes)')
plt.xticks(rotation=0)
plt.legend(["No Disease", "Has Disease"])
plt.ylabel('Frequency')
plt.show()

# 3. Fasting Blood Sugar (fbs) vs Target
pd.crosstab(df.fbs, df.target).plot(kind="bar", figsize=(10, 6), color=['#1CA53B','#AA1111'])
plt.title('Heart Disease Frequency for Fasting Blood Sugar')
plt.xlabel('Fasting Blood Sugar (0 = No, 1 = Yes)')
plt.xticks(rotation=0)
plt.legend(["No Disease", "Has Disease"])
plt.ylabel('Frequency')
plt.show()

# 4. Chest Pain Type (cp) vs Target
pd.crosstab(df.cp, df.target).plot(kind="bar", figsize=(12, 6), color=['#1CA53B','#AA1111'])
plt.title('Heart Disease Frequency for Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation=0)
plt.legend(["No Disease", "Has Disease"])
plt.ylabel('Frequency')
plt.show()

# 5. Correlation Matrix between all features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Heart Disease Dataset Features')
plt.show()

# 1. Prepare the features (X) and target (y)
y = df.target.values
X = df.drop(['target'], axis=1)

# 2. Scale the features (StandardScaler is commonly used to standardize the data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split the data into training and testing sets (30% test, 70% train)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. Initialize the Logistic Regression model
lr = LogisticRegression(max_iter=1000)

# 5. Train the model on the training data
lr.fit(X_train, y_train)

# 6. Calculate accuracy on the test data
y_pred = lr.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100

# 7. Store and print the accuracy
accuracies = {}
accuracies['Logistic Regression'] = acc
print(f"Test Accuracy: {accuracies['Logistic Regression']:.2f}%")

# 1. Prepare the features (X) and target (y)
y = df.target.values
X = df.drop(['target'], axis=1)

# 2. Scale the features (StandardScaler is commonly used to standardize the data)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split the data into training and testing sets (30% test, 70% train)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. Initialize the KNN model with k=3 (3 nearest neighbors)
knn = KNeighborsClassifier(n_neighbors=3)

# 5. Train the KNN model on the training data
knn.fit(X_train, y_train)

# 6. Make predictions on the test data
y_pred = knn.predict(X_test)

# 7. Calculate accuracy on the test data
acc = accuracy_score(y_test, y_pred) * 100

# 8. Print the model accuracy
print(f"{3} NN Score: {acc:.2f}%")

from tkinter import *
import tkinter as tk
df = pd.read_csv("heart.csv")



def check_inputs():
    if age.get() == "":
        print("Age Field is Empty!!")
        Label(win,text="Age Field is Empty!!",fg="blue",bg="yellow",font = ("Calibri 10 bold")).place(x=12,y=580)


    elif rbp.get() == "":
        print("Resting Blood Pressure Field is Empty!!")
        Label(win,text="Resting Blood Pressure Field is Empty!!",fg="blue",bg="yellow",font = ("Calibri 10 bold")).place(x=12,y=580)

    elif chol.get() == "":
        print("Cholestrol Field is Empty!!")
        Label(win,text="Cholestrol Field is Empty!!",fg="blue",bg="yellow",font = ("Calibri 10 bold")).place(x=12,y=580)

    elif heart_rate.get() == "":
        print("Heart Rate Field is Empty!!")
        
        Label(win,text="Heart Rate Field is Empty!!",fg="blue",bg="yellow",font = ("Calibri 10 bold")).place(x=12,y=580)
    elif peak.get() == "":
        print("Depression By Exercise Field is Empty!!")
        Label(win,text="Depression By Exercise Field is Empty!!",fg="blue",bg="yellow",font = ("Calibri 10 bold")).place(x=12,y=580)

    else:
        predict()


def predict():
    gender_dict = {"Male":1, "Female":0}
    fbs_dict = {"True":1, "False":0}
    eia_dict = {"True":1, "False":0}
    cp_dict = {"1: typical angina":0,"2: atypical angina":1,"3: non-anginal pain":2,"4: asymptomatic":3}
    thal_dict = {"0: No Test":0,"1: Fixed Defect":1,"2: Normal Flow":2,"3: Reversible Defect":3}
    Pred_dict = {0:"Prediction: No Heart Disease Detected", 1:"Prediction: Signs of Heart Disease Deteced\nYou should consult with your Doctor!"}
    
    data = [float(age.get()),gender_dict[str(radio.get())], cp_dict[str(variable.get())], float(rbp.get()),
           float(chol.get()),fbs_dict[str(radio_fbs.get())], int(str(variable_ecg.get())) - 1 , float(heart_rate.get()),
           eia_dict[str(radio_eia.get())], float(peak.get()), int(str(variable_slope.get())) - 1,int(str(variable_n_vessels.get())) - 1,
           thal_dict[str(variable_thal.get())]]

    prediction = Final_Model.predict(np.array(data).reshape(1,13))
    pred_label = Pred_dict[prediction.tolist()[0]]
    Label(win,text=pred_label,fg="blue",bg="yellow",font = ("Calibri 10 bold")).place(x=12,y=580)



def reset():
    age.set("")
    rbp.set("")
    chol.set("")
    heart_rate.set("")
    peak.set("")




win =  Tk()

win.geometry("450x600")
win.configure(background="#Eaedee")
win.title("Heart Disease Classifier")
# win.iconbitmap('icon.ico')

title = Label(win, text="Heart Disease Classifier", bg="#2583be", width="300", height="2", fg="white", font = ("Arial 20 italic")).pack()

age = Label(win, text="Age in Years", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=65)

rbp = Label(win, text="Resting Blood Pressure ", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=105)

chol = Label(win, text="Cholestrol mg/dl ", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=145)

heart_rate = Label(win, text="Maximum Heart Rate ", bg="#Eaedee", font=("Verdana 12")).place(x=12, y=185)

peak = Label(win, text="Depression By Exercise ",bg="#Eaedee",font = ("Verdana 12")).place(x=12,y=225)


  
Gender = Label(win, text="Gender ",bg="#Eaedee",font = ("Verdana 12")).place(x=12,y=265)

radio = StringVar()
Male = Radiobutton(win, text="Male",bg="#Eaedee",variable=radio,value="Male",font = ("Verdana 12")).place(x=160,y=265)
Female = Radiobutton(win, text="Female",bg="#Eaedee",variable=radio,value="Female",font = ("Verdana 12")).place(x=260,y=265)

FBS = Label(win, text="Fasting Blood Pressure ",bg="#Eaedee",font = ("Verdana 12")).place(x=12,y=285)

radio_fbs = StringVar()
Male = Radiobutton(win, text="True",bg="#Eaedee",variable=radio_fbs,value="True",font = ("Verdana 12")).place(x=160,y=285)
Female = Radiobutton(win, text="False",bg="#Eaedee",variable=radio_fbs,value="False",font = ("Verdana 12")).place(x=260,y=285)

EIA = Label(win, text="Exercise Induced Angina",bg="#Eaedee",font = ("Verdana 12")).place(x=12,y=305)

radio_eia = StringVar()
Male = Radiobutton(win, text="True",bg="#Eaedee",variable=radio_eia,value="True",font = ("Verdana 12")).place(x=160,y=305)
Female = Radiobutton(win, text="False",bg="#Eaedee",variable=radio_eia,value="False",font = ("Verdana 12")).place(x=260,y=305)


cp = Label(win,text="Chest Pain ",bg="#Eaedee",font = ("Verdana 12")).place(x=12,y=345)
variable = StringVar(win)
variable.set("CP")
w = OptionMenu(win, variable, "1: typical angina","2: atypical angina","3: non-anginal pain","4: asymptomatic")
w.place(x=140,y=345)

ecg = Label(win,text="Resting ECG ",bg="#Eaedee",font = ("Verdana 12")).place(x=12,y=385)
variable_ecg = StringVar(win)
variable_ecg.set("ECG")
w_ecg = OptionMenu(win, variable_ecg, "1","2","3")
w_ecg.place(x=140,y=385)


exer_slope = Label(win,text="Slope of Exercise ",bg="#Eaedee",font = ("Verdana 12")).place(x=12,y=425)
variable_slope = StringVar(win)
variable_slope.set("Slope")
w_slope = OptionMenu(win, variable_slope, "1","2","3")
w_slope.place(x=140,y=425)


thal_label = Label(win,text="Thallium Stress ",bg="#Eaedee",font = ("Verdana 12")).place(x=12,y=465)
variable_thal = StringVar(win)
variable_thal.set("Thal")
w_thal = OptionMenu(win, variable_thal, "0: No Test","1: Fixed Defect","2: Normal Flow","3: Reversible Defect")
w_thal.place(x=140,y=465)


n_vessels = Label(win,text="Number Vessels ",bg="#Eaedee",font = ("Verdana 12")).place(x=12,y=505)
variable_n_vessels = StringVar(win)
variable_n_vessels.set("N_Vessels")
w_n_vessels = OptionMenu(win, variable_n_vessels, "1","2","3","4")
w_n_vessels.place(x=140,y=505)


age = StringVar()
rbp = StringVar()
chol = StringVar()
heart_rate = StringVar()
peak  = StringVar()
Gender = StringVar()
FBS  = StringVar()
EIA  = StringVar()
cp  = StringVar()
ecg  = StringVar()
exer_slope  = StringVar()
thal_label  = StringVar()
n_vessels  = StringVar()

entry_age = Entry(win,textvariable = age,width=30)
entry_age.place(x=150,y=65)

entry_rbp = Entry(win,textvariable = rbp,width=30)
entry_rbp.place(x=150,y=105)

entry_chol = Entry(win,textvariable = chol,width=30)
entry_chol.place(x=150,y=145)

entry_heart_rate = Entry(win, textvariable = heart_rate,width=30)
entry_heart_rate.place(x=150,y=185)

entry_peak = Entry(win,textvariable = peak,width=30)
entry_peak.place(x=150,y=225)

reset = Button(win, text="Reset", width="12",height="1",activebackground="red",command=reset, bg="Pink",font = ("Calibri 12 ")).place(x=24, y=540)
submit = Button(win, text="Classify", width="12",height="1",activebackground="violet", bg="Pink",command=check_inputs,font = ("Calibri 12 ")).place(x=240, y=540)


win.mainloop()
