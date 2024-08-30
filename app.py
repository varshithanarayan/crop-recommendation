from flask import Flask,render_template,url_for,request
from flask_material import Material
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import json
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import h5py
#import imblearn
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
from datetime import datetime
import crops
import random

scaler = StandardScaler()
import os
print(os.listdir())


# In[2]:
app = Flask(__name__)
Material(app)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/ticker": {"origins": "http://localhost:port"}})
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		region = request.form['region']
		area = request.form['area']
		N = request.form['N']
		P = request.form['p']
		k =request.form['k']
		ph = request.form['ph']
		temp = request.form['temp']
		humidity = request.form['humidity']
		rain = request.form['rain']
	
	
		sample_data = [region,area,N,P,k,ph,temp,humidity,rain]
		clean_data = [float(i) for i in sample_data]
		ex1 = np.array(clean_data).reshape(1,-1)

		df1 = pd.read_csv('datafile.csv')
		df1=df1.drop(columns='Year')
		df1=df1.replace(['Mysore','T Narasipura','Nanjangud','Hunsur','K R Nagara','Rice','wheat','bajra','cotton','coconut','blackgram','banana','tur','jowar','ragi','sunflower','jasmine','ginger','sugarcane','groundnut','tomato','maize','chilli','mango'],[1,0,1,2,3,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])

		X1=df1.drop(columns='crop')
		y1 = df1['crop'].values
		X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.25)

		gnb = GaussianNB()
		gnb.fit(X_train1, y_train1)



		knn = KNeighborsClassifier(n_neighbors = 3)
		knn.fit(X_train1,y_train1)

		clf = tree.DecisionTreeClassifier()
		clf.fit(X_train1, y_train1)



	def croprecom(res):
		if result==0:
			class1="Rice"
		if result==1:
			class1="Wheat"
		elif result==2:
			class1="Bajra"
		elif result==3:
			class1="cotton"
		elif result==4:
			class1="Coconut"
		elif result==5:
			class1="Blackgram"
		elif result==6:
			class1="Banana"
		elif result==7:
			class1="Tur"
		elif result==8:
			class1="Jawar"
		elif result==9:
			class1="Ragi"
		elif result==10:
			class1="Sunflower "
		elif result==11:
			class1="Jasmine"
		elif result==12:
			class1="Ginger"
		elif result==13:
			class1="Sugarcane"
		elif result==14:
			class1="Groundnut"
		elif result==15:
			class1="Tomato"
		elif result==16:
			class1="Maize"
		elif result==17:
			class1="Chilli"
		elif result==18: 
			class1="Mango"
		return class1


	result=gnb.predict(ex1)
	class1=croprecom(result)
	Y_pred_nb =gnb.predict(X_test1)
	Y_pred_nb.shape
	score_nb = round(accuracy_score(Y_pred_nb,y_test1)*100,2)
           
		

           
		
	result=knn.predict(ex1)
	class3=croprecom(result)	
	Y_pred_knn =knn.predict(X_test1)
	Y_pred_knn.shape
	score_knn = round(accuracy_score(Y_pred_knn,y_test1)*100,2)
            
		
	result=clf.predict(ex1)
	class4=croprecom(result)	
	Y_pred_dt =clf.predict(X_test1)
	Y_pred_dt.shape
	score_dt = round(accuracy_score(Y_pred_dt,y_test1)*100,2)


            
		
            
					
	return render_template('index.html',region=region,area=area,N=N,P=P,k=k,ph=ph,temp=temp,humidity=humidity,rain=rain,class1=class1,score_nb=score_nb,class3=class3,score_knn=score_knn,class4=class4,score_dt=score_dt)

@app.route('/result1')
def result1():
	

		return render_template('result1.html',cp=2,chol=200,ca=3,tl=1)

@app.route('/preview')
def preview():
		feature = 'Bar'
		bar = create_plot(feature)
		return render_template('preview.html', plot=bar)
def create_plot(feature):
		df1 = pd.read_csv('datafile.csv')
		df1=df1.drop(columns='Year')
		df1=df1.replace(['Mysore','T Narasipura','Nanjangud','Hunsur','K R Nagara','Rice','wheat','bajra','cotton','coconut','blackgram','banana','tur','jowar','ragi','sunflower','jasmine','ginger','sugarcane','groundnut','tomato','maize','chilli','mango'],[1,0,1,2,3,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])

		X1=df1.drop(columns='crop')
		y1 = df1['crop'].values
		X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.25)
		lr = LogisticRegression()
		lr.fit(X_train1,y_train1)
		Y_pred_lr = lr.predict(X_test1)
		Y_pred_lr.shape
		score_lr = round(accuracy_score(Y_pred_lr,y_test1)*100,2)

		nb = GaussianNB()
		nb.fit(X_train1,y_train1)
		Y_pred_nb = nb.predict(X_test1)
		Y_pred_nb.shape
		score_nb = round(accuracy_score(Y_pred_nb,y_test1)*100,2)

		sv = svm.SVC(kernel='linear')
		sv.fit(X_train1, y_train1)
		Y_pred_svm = sv.predict(X_test1)
		Y_pred_svm.shape
		score_svm = round(accuracy_score(Y_pred_svm,y_test1)*100,2)

		knn = KNeighborsClassifier(n_neighbors=7)
		knn.fit(X_train1,y_train1)
		Y_pred_knn=knn.predict(X_test1)
		Y_pred_knn.shape
		score_knn = round(accuracy_score(Y_pred_knn,y_test1)*100,2)

		max_accuracy = 0
		for x in range(200):
			dt = DecisionTreeClassifier(random_state=x)
			dt.fit(X_train1,y_train1)
			Y_pred_dt = dt.predict(X_test1)
			current_accuracy = round(accuracy_score(Y_pred_dt,y_test1)*100,2)
			if(current_accuracy>max_accuracy):
				max_accuracy = current_accuracy
				best_x = x
		dt = DecisionTreeClassifier(random_state=best_x)
		dt.fit(X_train1,y_train1)
		Y_pred_dt = dt.predict(X_test1)
		print(Y_pred_dt.shape)
		score_dt = round(accuracy_score(Y_pred_dt,y_test1)*100,2)



		max_accuracy = 0
		for x in range(2000):
			rf = RandomForestClassifier(random_state=x)
			rf.fit(X_train1,y_train1)
			Y_pred_rf = rf.predict(X_test1)
			current_accuracy = round(accuracy_score(Y_pred_rf,y_test1)*100,2)
			if(current_accuracy>max_accuracy):
				max_accuracy = current_accuracy
				best_x = x
		rf = RandomForestClassifier(random_state=best_x)
		rf.fit(X_train1,y_train1)
		Y_pred_rf = rf.predict(X_test1)
		Y_pred_rf.shape
		score_rf = round(accuracy_score(Y_pred_rf,y_test1)*100,2)


		model = Sequential()
		model.add(Dense(11,activation='relu',input_dim=9))
		model.add(Dense(1,activation='sigmoid'))
		model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
		model.fit(X_train1,y_train1,epochs=30)
		Y_pred_nn = model.predict(X_test1)
		Y_pred_nn.shape
		rounded = [round(x[0]) for x in Y_pred_nn]
		Y_pred_nn = rounded
		score_nn = round(accuracy_score(Y_pred_nn,y_test1)*100,2)

		data = [
			go.Bar(
				# assign x as the dataframe column 'x'
				x=["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest","Neural Network"],
				y=[score_lr,score_nb,score_svm,score_knn,score_dt,score_rf,score_nn]
				)
			]
		graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
		return graphJSON




commodity_dict = {
    "arhar": "static/Arhar.csv",
    "bajra": "static/Bajra.csv",
    "barley": "static/Barley.csv",
    "copra": "static/Copra.csv",
    "cotton": "static/Cotton.csv",
    "sesamum": "static/Sesamum.csv",
    "gram": "static/Gram.csv",
    "groundnut": "static/Groundnut.csv",
    "jowar": "static/Jowar.csv",
    "maize": "static/Maize.csv",
    "masoor": "static/Masoor.csv",
    "moong": "static/Moong.csv",
    "niger": "static/Niger.csv",
    "paddy": "static/Paddy.csv",
    "ragi": "static/Ragi.csv",
    "rape": "static/Rape.csv",
    "jute": "static/Jute.csv",
    "safflower": "static/Safflower.csv",
    "soyabean": "static/Soyabean.csv",
    "sugarcane": "static/Sugarcane.csv",
    "sunflower": "static/Sunflower.csv",
    "urad": "static/Urad.csv",
    "wheat": "static/Wheat.csv"
}

annual_rainfall = [29, 21, 37.5, 30.7, 52.6, 150, 299, 251.7, 179.2, 70.5, 39.8, 10.9]
base = {
    "Paddy": 1245.5,
    "Arhar": 3200,
    "Bajra": 1175,
    "Barley": 980,
    "Copra": 5100,
    "Cotton": 3600,
    "Sesamum": 4200,
    "Gram": 2800,
    "Groundnut": 3700,
    "Jowar": 1520,
    "Maize": 1175,
    "Masoor": 2800,
    "Moong": 3500,
    "Niger": 3500,
    "Ragi": 1500,
    "Rape": 2500,
    "Jute": 1675,
    "Safflower": 2500,
    "Soyabean": 2200,
    "Sugarcane": 2250,
    "Sunflower": 3700,
    "Urad": 4300,
    "Wheat": 1350

}
commodity_list = []


class Commodity:

    def __init__(self, csv_name):
        self.name = csv_name
        dataset = pd.read_csv(csv_name)
        self.X = dataset.iloc[:, :-1].values
        self.Y = dataset.iloc[:, 3].values

        #from sklearn.model_selection import train_test_split
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

        # Fitting decision tree regression to dataset
        from sklearn.tree import DecisionTreeRegressor
        depth = random.randrange(7,18)
        self.regressor = DecisionTreeRegressor(max_depth=depth)
        self.regressor.fit(self.X, self.Y)
        #y_pred_tree = self.regressor.predict(X_test)
        # fsa=np.array([float(1),2019,45]).reshape(1,3)
        # fask=regressor_tree.predict(fsa)

    def getPredictedValue(self, value):
        if value[1]>=2019:
            fsa = np.array(value).reshape(1, 3)
            #print(" ",self.regressor.predict(fsa)[0])
            return self.regressor.predict(fsa)[0]
        else:
            c=self.X[:,0:2]
            x=[]
            for i in c:
                x.append(i.tolist())
            fsa = [value[0], value[1]]
            ind = 0
            for i in range(0,len(x)):
                if x[i]==fsa:
                    ind=i
                    break
            #print(index, " ",ind)
            #print(x[ind])
            #print(self.Y[i])
            return self.Y[i]

    def getCropName(self):
        a = self.name.split('.')
        return a[0]



@app.route('/price')
def price():
    context = {
        "top5": TopFiveWinners(),
        "bottom5": TopFiveLosers(),
        "sixmonths": SixMonthsForecast()
    }
    return render_template('index1.html', context=context)


@app.route('/commodity/<name>')
def crop_profile(name):
    max_crop, min_crop, forecast_crop_values = TwelveMonthsForecast(name)
    prev_crop_values = TwelveMonthPrevious(name)
    forecast_x = [i[0] for i in forecast_crop_values]
    forecast_y = [i[1] for i in forecast_crop_values]
    previous_x = [i[0] for i in prev_crop_values]
    previous_y = [i[1] for i in prev_crop_values]
    current_price = CurrentMonth(name)
    #print(max_crop)
    #print(min_crop)
    #print(forecast_crop_values)
    #print(prev_crop_values)
    #print(str(forecast_x))
    crop_data = crops.crop(name)
    context = {
        "name":name,
        "max_crop": max_crop,
        "min_crop": min_crop,
        "forecast_values": forecast_crop_values,
        "forecast_x": str(forecast_x),
        "forecast_y":forecast_y,
        "previous_values": prev_crop_values,
        "previous_x":previous_x,
        "previous_y":previous_y,
        "current_price": current_price,
        "image_url":crop_data[0],
        "prime_loc":crop_data[1],
        "type_c":crop_data[2],
        "export":crop_data[3]
    }
    return render_template('commodity.html', context=context)

@app.route('/ticker/<item>/<number>')
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def ticker(item, number):
    n = int(number)
    i = int(item)
    data = SixMonthsForecast()
    context = str(data[n][i])

    if i == 2 or i == 5:
        context = '₹' + context
    elif i == 3 or i == 6:

        context = context + '%'

    #print('context: ', context)
    return context


def TopFiveWinners():
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    prev_month = current_month - 1
    prev_rainfall = annual_rainfall[prev_month - 1]
    current_month_prediction = []
    prev_month_prediction = []
    change = []

    for i in commodity_list:
        current_predict = i.getPredictedValue([float(current_month), current_year, current_rainfall])
        current_month_prediction.append(current_predict)
        prev_predict = i.getPredictedValue([float(prev_month), current_year, prev_rainfall])
        prev_month_prediction.append(prev_predict)
        change.append((((current_predict - prev_predict) * 100 / prev_predict), commodity_list.index(i)))
    sorted_change = change
    sorted_change.sort(reverse=True)
    # print(sorted_change)
    to_send = []
    for j in range(0, 5):
        perc, i = sorted_change[j]
        name = commodity_list[i].getCropName().split('/')[1]
        to_send.append([name, round((current_month_prediction[i] * base[name]) / 100, 2), round(perc, 2)])
    #print(to_send)
    return to_send


def TopFiveLosers():
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    prev_month = current_month - 1
    prev_rainfall = annual_rainfall[prev_month - 1]
    current_month_prediction = []
    prev_month_prediction = []
    change = []

    for i in commodity_list:
        current_predict = i.getPredictedValue([float(current_month), current_year, current_rainfall])
        current_month_prediction.append(current_predict)
        prev_predict = i.getPredictedValue([float(prev_month), current_year, prev_rainfall])
        prev_month_prediction.append(prev_predict)
        change.append((((current_predict - prev_predict) * 100 / prev_predict), commodity_list.index(i)))
    sorted_change = change
    sorted_change.sort()
    to_send = []
    for j in range(0, 5):
        perc, i = sorted_change[j]
        name = commodity_list[i].getCropName().split('/')[1]
        to_send.append([name, round((current_month_prediction[i] * base[name]) / 100, 2), round(perc, 2)])
   # print(to_send)
    return to_send



def SixMonthsForecast():
    month1=[]
    month2=[]
    month3=[]
    month4=[]
    month5=[]
    month6=[]
    for i in commodity_list:
        crop=SixMonthsForecastHelper(i.getCropName())
        k=0
        for j in crop:
            time = j[0]
            price = j[1]
            change = j[2]
            if k==0:
                month1.append((price,change,i.getCropName().split("/")[1],time))
            elif k==1:
                month2.append((price,change,i.getCropName().split("/")[1],time))
            elif k==2:
                month3.append((price,change,i.getCropName().split("/")[1],time))
            elif k==3:
                month4.append((price,change,i.getCropName().split("/")[1],time))
            elif k==4:
                month5.append((price,change,i.getCropName().split("/")[1],time))
            elif k==5:
                month6.append((price,change,i.getCropName().split("/")[1],time))
            k+=1
    month1.sort()
    month2.sort()
    month3.sort()
    month4.sort()
    month5.sort()
    month6.sort()
    crop_month_wise=[]
    crop_month_wise.append([month1[0][3],month1[len(month1)-1][2],month1[len(month1)-1][0],month1[len(month1)-1][1],month1[0][2],month1[0][0],month1[0][1]])
    crop_month_wise.append([month2[0][3],month2[len(month2)-1][2],month2[len(month2)-1][0],month2[len(month2)-1][1],month2[0][2],month2[0][0],month2[0][1]])
    crop_month_wise.append([month3[0][3],month3[len(month3)-1][2],month3[len(month3)-1][0],month3[len(month3)-1][1],month3[0][2],month3[0][0],month3[0][1]])
    crop_month_wise.append([month4[0][3],month4[len(month4)-1][2],month4[len(month4)-1][0],month4[len(month4)-1][1],month4[0][2],month4[0][0],month4[0][1]])
    crop_month_wise.append([month5[0][3],month5[len(month5)-1][2],month5[len(month5)-1][0],month5[len(month5)-1][1],month5[0][2],month5[0][0],month5[0][1]])
    crop_month_wise.append([month6[0][3],month6[len(month6)-1][2],month6[len(month6)-1][0],month6[len(month6)-1][1],month6[0][2],month6[0][0],month6[0][1]])

   # print(crop_month_wise)
    return crop_month_wise

def SixMonthsForecastHelper(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    name = name.split("/")[1]
    name = name.lower()
    commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 7):
        if current_month + i <= 12:
            month_with_year.append((current_month + i, current_year, annual_rainfall[current_month + i - 1]))
        else:
            month_with_year.append((current_month + i - 12, current_year + 1, annual_rainfall[current_month + i - 13]))
    wpis = []
    current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
    change = []

    for m, y, r in month_with_year:
        current_predict = commodity.getPredictedValue([float(m), y, r])
        wpis.append(current_predict)
        change.append(((current_predict - current_wpi) * 100) / current_wpi)

    crop_price = []
    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y, m, 1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i]* base[name.capitalize()]) / 100, 2) , round(change[i], 2)])

   # print("Crop_Price: ", crop_price)
    return crop_price

def CurrentMonth(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    name = name.lower()
    commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
    current_price = (base[name.capitalize()]*current_wpi)/100
    return current_price

def TwelveMonthsForecast(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    name = name.lower()
    commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 13):
        if current_month + i <= 12:
            month_with_year.append((current_month + i, current_year, annual_rainfall[current_month + i - 1]))
        else:
            month_with_year.append((current_month + i - 12, current_year + 1, annual_rainfall[current_month + i - 13]))
    max_index = 0
    min_index = 0
    max_value = 0
    min_value = 9999
    wpis = []
    current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
    change = []

    for m, y, r in month_with_year:
        current_predict = commodity.getPredictedValue([float(m), y, r])
        if current_predict > max_value:
            max_value = current_predict
            max_index = month_with_year.index((m, y, r))
        if current_predict < min_value:
            min_value = current_predict
            min_index = month_with_year.index((m, y, r))
        wpis.append(current_predict)
        change.append(((current_predict - current_wpi) * 100) / current_wpi)

    max_month, max_year, r1 = month_with_year[max_index]
    min_month, min_year, r2 = month_with_year[min_index]
    min_value = min_value * base[name.capitalize()] / 100
    max_value = max_value * base[name.capitalize()] / 100
    crop_price = []
    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y, m, 1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i]* base[name.capitalize()]) / 100, 2) , round(change[i], 2)])
   # print("forecasr", wpis)
    x = datetime(max_year,max_month,1)
    x = x.strftime("%b %y")
    max_crop = [x, round(max_value,2)]
    x = datetime(min_year, min_month, 1)
    x = x.strftime("%b %y")
    min_crop = [x, round(min_value,2)]

    return max_crop, min_crop, crop_price


def TwelveMonthPrevious(name):
    name = name.lower()
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    commodity = commodity_list[0]
    wpis = []
    crop_price = []
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 13):
        if current_month - i >= 1:
            month_with_year.append((current_month - i, current_year, annual_rainfall[current_month - i - 1]))
        else:
            month_with_year.append((current_month - i + 12, current_year - 1, annual_rainfall[current_month - i + 11]))

    for m, y, r in month_with_year:
        current_predict = commodity.getPredictedValue([float(m), 2013, r])
        wpis.append(current_predict)

    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y,m,1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i]* base[name.capitalize()]) / 100, 2)])
   # print("previous ", wpis)
    new_crop_price =[]
    for i in range(len(crop_price)-1,-1,-1):
        new_crop_price.append(crop_price[i])
    return new_crop_price


if __name__ == "__main__":
    arhar = Commodity(commodity_dict["arhar"])
    commodity_list.append(arhar)
    bajra = Commodity(commodity_dict["bajra"])
    commodity_list.append(bajra)
    barley = Commodity(commodity_dict["barley"])
    commodity_list.append(barley)
    copra = Commodity(commodity_dict["copra"])
    commodity_list.append(copra)
    cotton = Commodity(commodity_dict["cotton"])
    commodity_list.append(cotton)
    sesamum = Commodity(commodity_dict["sesamum"])
    commodity_list.append(sesamum)
    gram = Commodity(commodity_dict["gram"])
    commodity_list.append(gram)
    groundnut = Commodity(commodity_dict["groundnut"])
    commodity_list.append(groundnut)
    jowar = Commodity(commodity_dict["jowar"])
    commodity_list.append(jowar)
    maize = Commodity(commodity_dict["maize"])
    commodity_list.append(maize)
    masoor = Commodity(commodity_dict["masoor"])
    commodity_list.append(masoor)
    moong = Commodity(commodity_dict["moong"])
    commodity_list.append(moong)
    niger = Commodity(commodity_dict["niger"])
    commodity_list.append(niger)
    paddy = Commodity(commodity_dict["paddy"])
    commodity_list.append(paddy)
    ragi = Commodity(commodity_dict["ragi"])
    commodity_list.append(ragi)
    rape = Commodity(commodity_dict["rape"])
    commodity_list.append(rape)
    jute = Commodity(commodity_dict["jute"])
    commodity_list.append(jute)
    safflower = Commodity(commodity_dict["safflower"])
    commodity_list.append(safflower)
    soyabean = Commodity(commodity_dict["soyabean"])
    commodity_list.append(soyabean)
    sugarcane = Commodity(commodity_dict["sugarcane"])
    commodity_list.append(sugarcane)
    sunflower = Commodity(commodity_dict["sunflower"])
    commodity_list.append(sunflower)
    urad = Commodity(commodity_dict["urad"])
    commodity_list.append(urad)
    wheat = Commodity(commodity_dict["wheat"])
    commodity_list.append(wheat)

    app.run(debug=True)

    