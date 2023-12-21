import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset=pd.read_csv(r"D:\NIT\DECEMBER\18 DEC(POLY..)\18th\emp_sal.csv")

X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
########################################
from sklearn.svm import SVR
re1=SVR()
re1.fit(X, y)

y_pred=re1.predict(X)

# predicto 
svr_model_pred =re1.predict([[6.5]])
svr_model_pred

##############################


from sklearn.svm import SVR
re2=SVR( kernel="linear")
re2.fit(X, y)

# predicto 
svr_model_pred2 =re2.predict([[6.5]])
svr_model_pred2

#####################################


from sklearn.svm import SVR
re3=SVR( kernel="poly")
re3.fit(X, y)

# predicto 
svr_model_pred3 =re3.predict([[6.5]])
svr_model_pred3


####################################



from sklearn.svm import SVR
re4=SVR( kernel="sigmoid")
re4.fit(X, y)

# predicto 
svr_model_pred4 =re4.predict([[6.5]])
svr_model_pred4

############################################




from sklearn.svm import SVR
re5=SVR( degree=3)
re5.fit(X, y)


# predicto 
svr_model_pred5 =re5.predict([[6.5]])
svr_model_pred5

############################################



from sklearn.svm import SVR
re8=SVR( degree=2)
re8.fit(X, y)


# predicto 
svr_model_pred8 =re8.predict([[6.5]])
svr_model_pred8

############################################




from sklearn.svm import SVR
re9=SVR( degree=2,kernel="linear")
re9.fit(X, y)


# predicto 
svr_model_pred9 =re9.predict([[6.5]])
svr_model_pred9

############################################


from sklearn.svm import SVR
re10=SVR( degree=6,kernel="linear")
re10.fit(X, y)


# predicto 
svr_model_pred10 =re10.predict([[6.5]])
svr_model_pred10

############################################





from sklearn.svm import SVR
re7=SVR( degree=3,kernel="poly")
re7.fit(X, y)


# predicto 
svr_model_pred7=re7.predict([[6.5]])
svr_model_pred7

############################################


from sklearn.svm import SVR
re6=SVR( degree=5,kernel="poly")
re6.fit(X, y)


# predicto 
svr_model_pred6 =re6.predict([[6.5]])
svr_model_pred6

############################################



from sklearn.svm import SVR
re11=SVR( degree=4,kernel="sigmoid")
re11.fit(X, y)


# predicto 
svr_model_pred11=re11.predict([[6.5]])
svr_model_pred11

############################################


from sklearn.svm import SVR
re12=SVR( degree=7,kernel="sigmoid")
re12.fit(X, y)


# predicto 
svr_model_pred12 =re12.predict([[6.5]])
svr_model_pred12

############################################


from sklearn.svm import SVR
re13=SVR( degree=3, gamma="auto")
re13.fit(X, y)


# predicto 
svr_model_pred13 =re13.predict([[6.5]])
svr_model_pred13
############################################


from sklearn.svm import SVR
re14=SVR( degree=6,kernel="linear",gamma="auto")
re14.fit(X, y)


# predicto 
svr_model_pred14 =re14.predict([[6.5]])
svr_model_pred14
############################################


from sklearn.svm import SVR
re15=SVR( degree=5,kernel="poly",gamma="auto")
re15.fit(X, y)


# predicto 
svr_model_pred15 =re15.predict([[6.5]])
svr_model_pred15
############################################



from sklearn.svm import SVR
re16=SVR( degree=5,kernel="sigmoid",gamma="auto")
re16.fit(X, y)


# predicto 
svr_model_pred16 =re16.predict([[6.5]])
svr_model_pred16

############################################



from sklearn.svm import SVR
re17=SVR( degree=3, gamma="auto",epsilon=0.2)
re17.fit(X, y)


# predicto 
svr_model_pred17 =re17.predict([[6.5]])
svr_model_pred17
############################################



from sklearn.svm import SVR
re18=SVR( degree=6,kernel="linear",gamma="auto",epsilon=0.2)
re18.fit(X, y)


# predicto 
svr_model_pred18 =re18.predict([[6.5]])
svr_model_pred18
############################################



from sklearn.svm import SVR
re19=SVR( degree=5,kernel="poly",gamma="auto",epsilon=0.2)
re19.fit(X, y)


# predicto 
svr_model_pred19 =re19.predict([[6.5]])
svr_model_pred19
############################################


from sklearn.svm import SVR
re20=SVR( degree=5,kernel="sigmoid",gamma="auto",epsilon=0.3)
re20.fit(X, y)


# predicto 
svr_model_pred20 =re20.predict([[6.5]])
svr_model_pred20


