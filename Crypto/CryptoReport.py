
# coding: utf-8

# # Automated Detailed Crypto Analysis 
# ### Project Members : Adam McMurchie & Allan Forbes

# ![texte](https://www.crypto-news.net/wp-content/uploads/2016/11/1217.sdt-news.jpg)

# ## Summary
# In 2017 from discussions on the success of crytpocurrencies, advantages and shortcomings a plan was devised to explore potential gaps and opportunities in the market. A strategy was devised initially on a high level basis of building a user friendly trading interface and eventually a back end that would take advantage of the interface to automate trading.  
#  The Front End would become multiplatform both desktop and mobile, and the backend would employ machine learning to take advantage of recent developments in Neural Networks to optimise trading returns. 
# 
#  Significant progress has been made by Allan in this area, and both a prototype as well as an initial release of the front end has already been completed and can be sourced here https://github.com/doubleelforbes 
#  Whilst the front end will continue to be improved and further iterations released - work will now begin on designing and building the back end automation. 
# 

# ## Backend
# 
# The aim of this project is to build an automated predictive and reactive custom built programatic solution for selecting optimal trading pairs based upon historical and current crypto data as well as user balance and limitations (i.e. base pairs, SWOT analysis and profit margin). 
# 
# ## Analysis
# This Document will provide all the necessary code and design for obtaining metrics that will be required for the back end to initiate trading. This project is designed to be built upon incrementally and is a living breathing document. This means the initial goals will be basic objectives like, aggrigate data and sort into useful batches.
# ### Initial Target Coin : FTC Feathers
# For the purposes of building up a frame work to be used for all coins, we will start with one coin and work with data brought back from cryptopia API via Allan's front end and process data for history, probabilities, predictions etc.  
# FTC is currently 2500-3000 satoshi but has been as low as 900-1200 in the past.
# FTC has LTC and DOGE pairs and they bounce a lot LESS than the BTC price.
# This would be a good template to work from.
# 
# ## PROGRESS
# 
#  From the preliminary Analysis notebook we have been able to pull values for one coin FTC and report on several metrics. Theb biggest challenge was passing the 2D array into a list that allows for drilling down to specific values but high level enough for us to pick a given market  

# # DEPENDENCIES
# 

# In[1]:

# ======================== Python dependencies
import json
import pandas
import statistics
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os


# ======================== Self dependencies
from utilities import pubcapi
from utilities import coin
from utilities import market
# ======================== Functions


# ### Collate File Names To Array

# In[2]:

directory = "data"
selectedcoins = []

for filename in os.listdir(directory):
    if filename.endswith(".txt"): 
        #print(os.path.join(directory, filename))
        marketName = filename[:-4]
        selectedcoins.append(marketName)
        continue
    else:
        continue


# In[3]:

#initialize iterators
i = 0
j = 0
newRow = []
allFiles = []
targetArray = []



for j in range(0, len(selectedcoins)): # Iterate through files
    with open("data/" + str(selectedcoins[j]) + ".txt", 'rU') as f:
        
        
        newRow = []
        for line in f: # Iterate through rows
            line = line[:-1]
            words = line.split(",")
            # for each word in the line:
            #print(words)
            myarray = np.asarray(words)
            newRow.append(myarray)
        #print(newRow)
        targetArray.append(np.asarray(newRow))
        #for x in range (0, len(newRow)): print(targetArray[j][x])
        #for x in range (0, len(newRow)): print(str(targetArray[j][x][0]) + " " + str(targetArray[j][x][3]) + " " + str(targetArray[j][x][-1]))
        print("the number of records are: " + str(len(newRow)))
        print("the value of j is : " + str(j))
        

 


# In[4]:

len(targetArray[j])


# In[5]:

#  DATA PRINTOUT 

#------------------------------------GLOBAL PARAMETER SECTION------------------------------

tempA, tempB, tempC, tempD, tempE   = [],[],[],[],[] 

BS = []
VOL = []
RATES = [] # i hold rates [y] for each market [x]  rates[x][y]
TOTAL = []
DATES = []
   

    
minRate = []   
maxRate = []
avgRate = []
medRate = []
lastRatesX,lastRatesY,lastRatesZ = [],[],[]

minVol = []   
maxVol = []
avgVol = []
medVol = []
totalVol = []


totalBS = []
boughtBS = []
soldBS = []

print("MARKET DATA REPORT" + "\r\n")
print("DATE OF REPORT : " + str(datetime.datetime.now()) + "\r\n")
print("PROPERTY OF Adam McMurchie and Allan Forbes" + "\r\n")
print("Permission is required before distributing or sourcing" + "\r\n" + "\r\n" + "\r\n" + "\r\n" )




for j in range (0, len(selectedcoins)):
    
    #------------------------------------PARAMETERIZE DATA SECTION ------------------------------
    
    #extract the rates to a 2D array
    for x in range(0, len(targetArray[j])):
        tempA.append(targetArray[j][x][1])        # BUY OR SELL
        tempB.append(float(targetArray[j][x][2])) # VOLUME
        tempC.append(float(targetArray[j][x][3])) # RATE
        tempD.append(float(targetArray[j][x][4])) # TOTAL
        #tempE.append(targetArray[j][x][5])        # DATE
        toDatetime = datetime.datetime.strptime(targetArray[j][x][5], "%Y-%m-%d %H:%M:%S")
        tempE.append(toDatetime)        # DATE
    
    BS.append(np.asarray(tempA))
    VOL.append(np.asarray(tempB))
    RATES.append(np.asarray(tempC))
    TOTAL.append(np.asarray(tempD))
    DATES.append(np.asarray(tempE))
    
    tempA, tempB, tempC, tempD, tempE   = [],[],[],[],[]
    
    #---------------------------------PROCESS DATA SECTION --------------------------------------
    # DATES
    mindate = min(DATES[j])
    maxdate = max(DATES[j])
    #newmindate = datetime.datetime.strptime(mindate, "%Y-%m-%d %H:%M:%S")
    #newmaxdate = datetime.datetime.strptime(maxdate, "%Y-%m-%d %H:%M:%S")
    timeduration = maxdate - mindate
    
    # RATES 
    #Original way maxRate = max(RATES[j])
    minRate.append(min(RATES[j]))
    maxRate.append(max(RATES[j]))
    avgRate.append(sum(RATES[j])/len(RATES[j]))
    medRate.append(statistics.median(RATES[j]))
    lastRatesX.append(RATES[j][-3])
    lastRatesY.append(RATES[j][-2])
    lastRatesZ.append(RATES[j][-1])
    
    averageRateGraph = np.repeat((avgRate[j]), len(targetArray[j]))
    medianRateGraph = np.repeat((medRate[j]), len(targetArray[j])) 
    
    
    # VOLUMES 
    #Original way maxRate = max(VOL[j])
    minVol.append(min(VOL[j]))
    maxVol.append(max(VOL[j]))
    avgVol.append(sum(VOL[j])/len(VOL[j]))
    medVol.append(statistics.median(VOL[j]))
    totalVol.append(sum(VOL[j]))
    
    averageVolGraph = np.repeat((avgVol[j]), len(targetArray[j]))
    medianVolGraph = np.repeat((medVol[j]), len(targetArray[j]))

    # Buy Sell 
    totalBS.append(len(BS[j]))
    boughtBS.append(str(BS[j]).count('Buy'))
    soldBS.append(str(BS[j]).count('Sell'))

    
    
    
    
    
    #---------------------------------DISPLAY DATA SECTION --------------------------------------
    
    print("****************     THIS IS FOR " + selectedcoins[j] + " MARKET     ****************" + "\r\n")
    print("The numer of rows are: " + str(len(targetArray[j])) + "\r\n")
    print("------Dates------- \n\r")
    print("The dates ranges are : " + str(mindate) + " TO " + str(maxdate) + "\r\n")
    #print(DATES[j])
    print("Time duraiton is: " + str(timeduration) + "\r\n")
    
    
    # PRINT RATE VALUES
    
    print("------RATES------- \n\r")
    print("Minimum and Maximum rates are : " + str("{0:.8f}".format(minRate[j])) + " " + str("{0:.8f}".format(maxRate[j])))
    print("Median and Average rates are : "  + str("{0:.8f}".format(medRate[j])) + " " + str("{0:.8f}".format(avgRate[j])))
    print("The last three rates are : "  + str("{0:.8f}".format(lastRatesX[j])) + " " + str("{0:.8f}".format(lastRatesY[j])) + " " + str("{0:.8f}".format(lastRatesZ[j])))
    
    # PLOT RATE VALUES
    
    plt.plot(RATES[j])
    plt.title(str(selectedcoins[j]) + " RATES")
    #plt.gca().invert_xaxis()
    plt.plot(averageRateGraph)
    plt.plot(medianRateGraph, 'r--')
    plt.ylabel(selectedcoins[j] + ' Rate')
    plt.xlabel(str(mindate) + "       Dates           " +  str(maxdate))
    print("Time duraiton is: " + str(timeduration))
    plt.show()
    
    
    # PRINT RATE VALUES

    
    print("\n\r")
    
    print("------VOLUMES------ \n\r")
    print("Total Trade Volume is : " + str("{0:.8f}".format(totalVol[j])))
    print("Minimum and Maximum Volumes are : " + str("{0:.8f}".format(minVol[j])) + " " + str("{0:.8f}".format(maxVol[j])))
    print("Median and Average Volumes are : "  + str("{0:.8f}".format(medVol[j])) + " " + str("{0:.8f}".format(avgVol[j])))

    # PLOT VOLUME VALUES
    
    ymax = 6 * avgVol[j]
    
    plt.plot(VOL[j],linestyle="",marker="o")
    #plt.gca().invert_xaxis()
    plt.gca().set_ylim([minVol[j],ymax])# set axis 
    plt.plot(averageVolGraph)
    plt.plot(medianVolGraph, 'r--')
    plt.title(str(selectedcoins[j]) + " Volumes")
    plt.ylabel(selectedcoins[j] + ' Volume')
    plt.xlabel(str(mindate) + "       Dates           " +  str(maxdate))
    print("Time duraiton is: " + str(timeduration))
    plt.show()

    print("\n\r")
    print("------BOUGHT AND SOLD------ \n\r")
    print("Total amount traded is : " + str("{0:.8f}".format(totalBS[j])))
    print("Total amount Bought : " + str("{0:.8f}".format(boughtBS[j])))
    print("Total amount Sold : " + str("{0:.8f}".format(soldBS[j])))
    
    
    plt.title(str(selectedcoins[j]) + 'bought versus Sold')
    objects = (str(selectedcoins[j][:3] + ' bought'), str(selectedcoins[j][:3] + 'sold'))
    y_pos = np.arange(len(objects))
    performance = [boughtBS[j],soldBS[j]]
    plt.bar(y_pos, performance, align='center', alpha=0.7)
    plt.xticks(y_pos, objects)
    plt.ylabel('Usage')
    plt.show()

    
    
    print("\n\r")




# In[6]:

"""import matplotlib
dates = matplotlib.dates.date2num(DATES[j])
matplotlib.pyplot.plot_date(dates, VOL[j])
plt.show()
"""


# In[7]:

# Produce an array of pairs but maintain index state (rows in current align to rows in target array)
current = []
for x in range(0, len(targetArray)):
    current.append(str(targetArray[x][0][0]).split("/")) # split coin1 and basecoin elements    
current


chosenCoinIndex = 40
chosenCoinPair = current[chosenCoinIndex]# current is list of pairs, ftc/btc, ftc/ltc etc 
print("finding paths for " + str(chosenCoinPair[0])) #only shows the selected coin from pair list
for x in range(0,len(current)):
    
    if chosenCoinPair[0] == current[x][0]:
        #print(("Available Bases are " + str(current[x][1]))) fix due to position in loop
        continue
    #print(current[x])
    if chosenCoinPair[1] == current[x][1]:
        print("PATH FOUND " + str(current[x][0]))
        print("Available via " + str(current[x][1]) + " swap")
        print("The " + str(current[x])  + " median rate is " + str(medRate[x]))
        print("The " + str(chosenCoinPair) + " median rate is " + str(medRate[chosenCoinIndex]))
        print("\r\n")


# ### NOTES 
# 
# 
# ## TRACKING TRANSACTIONS
# Sold BTC for DASH at 0.08149997 median was 0.0818 ish 
# 
# 
# 
# ### OUTSTANDING WORK
# throttle by changing this section for x in range(0, len(targetArray[j])):
# 
# len(targetArray[j]) is the list lenght 
# 
# for x in range((len(targetArray[j]) - 200 or something) , len(targetArray[j])):
# 
# 
# 
# 

# In[8]:

lastRatesY[j]


# In[ ]:



