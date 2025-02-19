import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter
pd.set_option('future.no_silent_downcasting', True)


def gradient(weights, featureMatrix, labels, dataNum):
    sum=np.zeros(weights.shape,dtype=float)
    for i in range(dataNum):
       
        label = np.float64(labels[i])
        feature = featureMatrix[i].reshape(-1, 1) 
        

        dotP = np.dot(weights, feature)[0][0] 
        

        denom = 1 + np.exp(label * dotP)
        num = -label * feature.T 
        
        out = num / denom
        sum = sum + out

    return sum/dataNum



def logisticRegression(weights, featureMatrix, labels, learningRate, epochs,dataNum):

    for i in range(epochs):
        print("Epoch: ", i)
        weights = weights - learningRate * gradient(weights, featureMatrix, labels,dataNum)
    return weights

def predict(xTest,yTest,newWeights):
    featureMatrix = np.hstack((xTest, np.ones((xTest.shape[0],1))))
    print(featureMatrix.shape,yTest.shape,newWeights.shape)
    correct = 0
    outArray=[]
    for i in range(featureMatrix.shape[0]):
        testY = (newWeights @ featureMatrix[i])
        outArray.append(1 if testY>=0 else -1)
        if (testY) > 0:
            if yTest[i] == 1:
                correct+=1
        else:
            if yTest[i] == -1:
                correct+=1
    
    
    print(Counter(outArray))
    return correct/len(outArray)


print("Loading dataset")
bigDF = pd.read_excel("LargerData.xlsx")


dataSet = bigDF[['State','Sex','AgeCategory','HeightInMeters','WeightInKilograms',
                 'BMI','HadAngina','HadStroke','HadAsthma','HadSkinCancer','HadCOPD',
                 'HadDiabetes','DifficultyWalking','SmokerStatus','AlcoholDrinkers',
                 'HighRiskLastYear','HadHeartAttack']]


states = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware',
                                             'Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',
                                             'Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska',
                                             'Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina','North Dakota','Ohio','Oklahoma',
                                             'Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virginia',
                                             'Washington','West Virginia','Wisconsin','Wyoming', "Puerto Rico", "District of Columbia", 'Guam', 'Virgin Islands']


dataSet['State'] = dataSet['State'].replace(states,list(range(len(states))))

ageCats = [
    'Age 65 to 69',     
'Age 60 to 64',
'Age 70 to 74',
'Age 55 to 59',
'Age 50 to 54',
'Age 75 to 79',
'Age 80 or older',
'Age 40 to 44',
'Age 45 to 49',
'Age 35 to 39',
'Age 30 to 34',
'Age 18 to 24',
'Age 25 to 29',
]
dataSet['AgeCategory'] = dataSet['AgeCategory'].replace(sorted(ageCats),list(range(len(ageCats))))

diab = [
    'No',
'Yes',
'No, pre-diabetes or borderline diabetes',
'Yes, but only during pregnancy (female)',
]

dataSet['HadDiabetes'] = dataSet['HadDiabetes'].replace(diab,[0,1,2,3])

smoke=[
    'Never smoked',
'Former smoker',
'Current smoker - now smokes every day',
'Current smoker - now smokes some days'
]

dataSet['SmokerStatus'] = dataSet['SmokerStatus'].replace(smoke,[0,1,2,3])
dataSet['Sex'] = dataSet['Sex'].replace(['Male','Female'],[0,1])
dataSet['HadHeartAttack'] = dataSet['HadHeartAttack'].replace([0,1],[-1,1])


bigRatio=0.8
dataSet = dataSet.sample(frac=1)
partition = int(bigRatio*len(dataSet))
Train, Test = dataSet.iloc[:partition],dataSet.iloc[partition:]

bigxData = np.array(Train.iloc[:,:-1])
bigyData = np.array(Train.iloc[:,-1])
bigxTest = np.array(Test.iloc[:,:-1])
bigyTest = np.array(Test.iloc[:,-1])
bigFeature = np.hstack((bigxData, np.ones((bigxData.shape[0],1))))

bigdataNum = bigFeature.shape[0]


weights = np.random.random((1,bigFeature.shape[1]))#(1,14)
correctedWeights = logisticRegression(weights, bigFeature,bigyData,10**-3,10,len(bigxData))
print("Accuracy: ",predict(bigxTest, bigyTest,correctedWeights))