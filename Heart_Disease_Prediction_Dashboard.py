from sklearn.linear_model import LogisticRegression
from tkinter import *
from sklearn import svm
from tkinter.ttk import Treeview
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from tkinter.messagebox import askyesno
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"D:\datasets\hospital (3).csv")
df.drop_duplicates(inplace=True)
df = shuffle(df)

x = df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg',
        'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']].values
y = df['output'].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
###############################################################################################################
# KNN
model_knn = KNeighborsClassifier(n_neighbors=7)
model_knn.fit(x_train, y_train)

y_pred_knn = model_knn.predict(x_test)
acc_knn = round(accuracy_score(y_test, y_pred_knn), 2)
recall_knn = round(recall_score(y_test, y_pred_knn), 2)
precision_knn = round(precision_score(y_test, y_pred_knn), 2)
f1_knn = round(f1_score(y_test, y_pred_knn), 2)


def knn_click():
    x_test_input = np.array([[entry_age.get(), entry_sex.get(), entry_cp.get(), entry_trtbps.get(), entry_chol.get(),
                              entry_fbs.get(), entry_restecg.get(), entry_thalachh.get(), entry_exng.get(),
                              entry_oldpeak.get(), entry_slp.get(), entry_caa.get(), entry_thall.get()]]).astype(np.float64)
    x_input_scaled = scaler.transform(x_test_input)
    prediction = model_knn.predict(x_input_scaled)
    f = entry_fullname.get()
    table_predict.insert('', END, values=(
        f, prediction[0], acc_knn, recall_knn, precision_knn, f1_knn, 'KNN'))
###########################################################################################################################


# SVM
model_svm = svm.SVC(kernel='rbf')
model_svm.fit(x_train, y_train)
y_pred_svm = model_svm.predict(x_test)
acc_svm = round(accuracy_score(y_test, y_pred_svm), 2)
recall_svm = round(recall_score(y_test, y_pred_svm), 2)
precision_svm = round(precision_score(y_test, y_pred_svm), 2)
f1_svm = round(f1_score(y_test, y_pred_svm), 2)


def svm_click():
    x_test_input = np.array([[entry_age.get(), entry_sex.get(), entry_cp.get(), entry_trtbps.get(), entry_chol.get(),
                              entry_fbs.get(), entry_restecg.get(), entry_thalachh.get(), entry_exng.get(),
                              entry_oldpeak.get(), entry_slp.get(), entry_caa.get(), entry_thall.get()]]).astype(np.float64)
    x_input_scaled = scaler.transform(x_test_input)
    prediction = model_svm.predict(x_input_scaled)

    f = entry_fullname.get()
    table_predict.insert('', END, values=(
        f, prediction[0], acc_svm, recall_svm, precision_svm, f1_svm, 'SVM'))
############################################################################################################


# Decision Tree
model_dt = DecisionTreeClassifier(criterion='entropy', max_depth=5)
model_dt.fit(x_train, y_train)
y_pred_dt = model_dt.predict(x_test)
acc_dt = round(accuracy_score(y_test, y_pred_dt), 2)
recall_dt = round(recall_score(y_test, y_pred_dt), 2)
precision_dt = round(precision_score(y_test, y_pred_dt), 2)
f1_dt = round(f1_score(y_test, y_pred_dt), 2)


def decision_tree_click():
    x_test_input = np.array([[entry_age.get(), entry_sex.get(), entry_cp.get(), entry_trtbps.get(), entry_chol.get(),
                              entry_fbs.get(), entry_restecg.get(), entry_thalachh.get(), entry_exng.get(),
                              entry_oldpeak.get(), entry_slp.get(), entry_caa.get(), entry_thall.get()]]).astype(np.float64)
    x_input_scaled = scaler.transform(x_test_input)
    prediction = model_dt.predict(x_input_scaled)
    f = entry_fullname.get()
    table_predict.insert('', END, values=(
        f, prediction[0], acc_dt, recall_dt, precision_dt, f1_dt, 'Decision Tree'))


#############################################################################################################
# Random Forest
model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(x_train, y_train)
y_pred_rf = model_rf.predict(x_test)
acc_rf = round(accuracy_score(y_test, y_pred_rf), 2)
recall_rf = round(recall_score(y_test, y_pred_rf), 2)
precision_rf = round(precision_score(y_test, y_pred_rf), 2)
f1_rf = round(f1_score(y_test, y_pred_rf), 2)


def random_forest_click():
    x_test_input = np.array([[entry_age.get(), entry_sex.get(), entry_cp.get(), entry_trtbps.get(), entry_chol.get(),
                              entry_fbs.get(), entry_restecg.get(), entry_thalachh.get(), entry_exng.get(),
                              entry_oldpeak.get(), entry_slp.get(), entry_caa.get(), entry_thall.get()]]).astype(np.float64)
    x_input_scaled = scaler.transform(x_test_input)
    prediction = model_rf.predict(x_input_scaled)
    f = entry_fullname.get()
    table_predict.insert('', END, values=(
        f, prediction[0], acc_rf, recall_rf, precision_rf, f1_rf, 'Random Forest'))


#############################################################################################################
# Logistic  Regression
model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)
y_pred_lr = model_lr.predict(x_test)
acc_lr = round(accuracy_score(y_test, y_pred_lr), 2)
recall_lr = round(recall_score(y_test, y_pred_lr), 2)
precision_lr = round(precision_score(y_test, y_pred_lr), 2)
f1_lr = round(f1_score(y_test, y_pred_lr), 2)


def logistic_regression_click():

    x_test_input = np.array([[entry_age.get(), entry_sex.get(), entry_cp.get(), entry_trtbps.get(), entry_chol.get(),
                              entry_fbs.get(), entry_restecg.get(), entry_thalachh.get(), entry_exng.get(),
                              entry_oldpeak.get(), entry_slp.get(), entry_caa.get(), entry_thall.get()]]).astype(np.float64)

    x_input_scaled = scaler.transform(x_test_input)
    prediction = model_lr.predict(x_input_scaled)
    table_predict.insert('', END, values=(entry_fullname.get(
    ), prediction[0], acc_lr, recall_lr, precision_lr, f1_lr, 'LogisticRegression'))


def close_click():
    if askyesno('exit', 'are you sure?'):
        root.destroy()


def clear():
    pass


root = Tk()
root.title('Heart Disease Prediction')
root.geometry('960x540')

lbl_fullname = Label(root, text='Full Name')
lbl_fullname.place(x=20, y=20)
entry_fullname = Entry(root)
entry_fullname.place(x=100, y=20)

lbl_code = Label(root, text='Code')
lbl_code.place(x=20, y=60)
entry_code = Entry(root)
entry_code.place(x=100, y=60)

lbl_age = Label(root, text='Age')
lbl_age.place(x=20, y=100)
entry_age = Entry(root)
entry_age.place(x=100, y=100)

lbl_sex = Label(root, text='Sex')
lbl_sex.place(x=20, y=140)
entry_sex = Entry(root)
entry_sex.place(x=100, y=140)

lbl_cp = Label(root, text='CP')
lbl_cp.place(x=20, y=180)
entry_cp = Entry(root)
entry_cp.place(x=100, y=180)

lbl_trtbps = Label(root, text='TRT BPS')
lbl_trtbps.place(x=250, y=20)
entry_trtbps = Entry(root)
entry_trtbps.place(x=350, y=20)

lbl_chol = Label(root, text='Chol')
lbl_chol.place(x=250, y=60)
entry_chol = Entry(root)
entry_chol.place(x=350, y=60)

lbl_fbs = Label(root, text='FBS')
lbl_fbs.place(x=250, y=100)
entry_fbs = Entry(root)
entry_fbs.place(x=350, y=100)

lbl_restecg = Label(root, text='Rest ECG')
lbl_restecg.place(x=250, y=140)
entry_restecg = Entry(root)
entry_restecg.place(x=350, y=140)

lbl_thalachh = Label(root, text='Thalachh')
lbl_thalachh.place(x=250, y=180)
entry_thalachh = Entry(root)
entry_thalachh.place(x=350, y=180)

lbl_exng = Label(root, text='Ex NG')
lbl_exng.place(x=480, y=20)
entry_exng = Entry(root)
entry_exng.place(x=580, y=20)

lbl_oldpeak = Label(root, text='Old Peak')
lbl_oldpeak.place(x=480, y=60)
entry_oldpeak = Entry(root)
entry_oldpeak.place(x=580, y=60)

lbl_slp = Label(root, text='SLP')
lbl_slp.place(x=480, y=100)
entry_slp = Entry(root)
entry_slp.place(x=580, y=100)

lbl_caa = Label(root, text='CAA')
lbl_caa.place(x=480, y=140)
entry_caa = Entry(root)
entry_caa.place(x=580, y=140)

lbl_thall = Label(root, text='Thall')
lbl_thall.place(x=480, y=180)
entry_thall = Entry(root)
entry_thall.place(x=580, y=180)

lbl_output = Label(root, text='Output')
lbl_output.place(x=20, y=220)
entry_output = Entry(root)
entry_output.place(x=100, y=220)


btn_knn = Button(root, text='KNN', command=knn_click, width=20)
btn_knn.place(x=750, y=20)

btn_svm = Button(root, text='SVM', command=svm_click, width=20)
btn_svm.place(x=750, y=60)

btn_dt = Button(root, text='Decision Tree',
                command=decision_tree_click, width=20)
btn_dt.place(x=750, y=100)

btn_rf = Button(root, text='Random Forest',
                command=random_forest_click, width=20)
btn_rf.place(x=750, y=140)

btn_clear = Button(root, text='LogisticRegression',
                   command=logistic_regression_click, width=20)
btn_clear.place(x=750, y=180)

btn_rf = Button(root, text='close',
                command=close_click, width=20)
btn_rf.place(x=750, y=220)


table_predict = Treeview(root, columns=(1, 2, 3, 4, 5, 6, 7), show='headings')
table_predict.place(x=20, y=280)

table_predict.column(1, width=100)
table_predict.heading(1, text='Full Name')

table_predict.column(2, width=100)
table_predict.heading(2, text='Predict')

table_predict.column(3, width=100)
table_predict.heading(3, text='Accuracy Score')

table_predict.column(4, width=100)
table_predict.heading(4, text='Recall Score')

table_predict.column(5, width=100)
table_predict.heading(5, text='Precision Score')

table_predict.column(6, width=100)
table_predict.heading(6, text='F1 Score')

table_predict.column(7, width=100)
table_predict.heading(7, text='Algorithm')

# table_predict.bind('<ButtonRelease>', select_item)

root.mainloop()
