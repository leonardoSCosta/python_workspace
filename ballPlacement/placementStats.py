import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy.stats as stats

def placeStats(fileName):
    try:
        dfTeam = pd.read_csv(fileName)
        # dfTeam.set_index(fileName.split('_').pop(0), inplace=True)
    except:
        print(".csv file not found")
        return
    
    dfTeam['Timestamp'] = dfTeam['Timestamp'].add(-dfTeam['Timestamp'].min())
    dfTeam.sort_values(by=['Timestamp'], inplace=True)

    dist = [distance.euclidean(a, b) for a,b in zip(dfTeam.loc[:,['BallX','BallY']].values, dfTeam.loc[:,['PlaceX','PlaceY']].values)]
    dfTeam['BallDistPlace'] = dist

    # trying to remove points that are probably not the ball
    noise = dfTeam.loc[:, 'BallDistPlace'] > 5.5
    dfTeam.drop(dfTeam.iloc[list(noise),:].index, inplace=True)
    dfTeam.reset_index(inplace=True)
    dfTeam.set_index(dfTeam.columns[0])

    print(dfTeam.head())
    # print(dfTeam.shape)
    # dfTeam.loc[:,['BallDistPlace']].plot(kind="line")
    # plt.show()

    dfTeamFiltered = pd.DataFrame()
    dfAux = pd.DataFrame()

    # placementTime, ballTraveledDist, ballPlacePointInitDist, validPlacement
    placementData = [[],[],[],[]]

    # print(dfTeam.head())
    # input()

    # ax = dfTeam.plot(kind='scatter', x='BallX', y='BallY', color='red')

    initIndex = 0
    for index in range(0,len(dfTeam['Timestamp'])-1):
        if dfTeam.loc[index+1,'Timestamp'] - dfTeam.loc[index,'Timestamp'] > 20e-3 or index+1 == len(dfTeam['Timestamp'])-1:
        # if distance.euclidean(list(dfTeam.loc[index+1,['PlaceX','PlaceY']]), list(dfTeam.loc[index,['PlaceX','PlaceY']])) > 0.1:
            dfAux = dfTeam.iloc[initIndex:index,:].copy()
            # print(dfAux.tail())
            # print(dfAux.shape)
            initIndex = index+1

            # ax = dfAux.plot(kind='scatter', x='PlaceX', y='PlaceY', color='yellow', s=150)
            # placeCircle = plt.Circle((dfAux.loc[dfAux['PlaceX'].last_valid_index(), ['PlaceX','PlaceY']].values), 0.15, fill=False)
            # ax.add_artist(placeCircle)
            # dfAux.plot(kind='scatter', x='BallX', y='BallY', color='blue', ax=ax)
            # print(dfAux.describe())
            # plt.show()

            removed = dfAux['BallX'].between(dfAux['BallX'].mean()-4, dfAux['BallX'].mean()+4)
            indexes = dfAux.iloc[list(~removed),:].index
            dfAux.drop(indexes, inplace=True, errors='ignore')

            removed = dfAux['BallY'].between(dfAux['BallY'].mean()-4, dfAux['BallY'].mean()+4)
            indexes = dfAux.iloc[list(~removed),:].index
            dfAux.drop(indexes, inplace=True, errors='ignore')

            ballTravDist = 0
            indexList = list(dfAux.index)
            indexList.pop(0)

            dists = distance.cdist(list(dfAux.loc[:, ['BallX','BallY']].values), list(dfAux.loc[:, ['BallX','BallY']].values))

            for index in range(1,len(dists)):
                ballTravDist += dists[index-1][index]

            placementData[0].append(dfAux.loc[dfAux.last_valid_index(), 'Timestamp'] - dfAux.loc[dfAux.first_valid_index(), 'Timestamp'])
            placementData[1].append(ballTravDist)
            placementData[2].append(dfAux.loc[dfAux.first_valid_index(), 'BallDistPlace'])

            if distance.euclidean(list(dfAux.loc[dfAux.last_valid_index(), ['BallX','BallY']].values), \
                                  list(dfAux.loc[dfAux.last_valid_index(), ['PlaceX','PlaceY']].values)) <= 0.15:
                placementData[3].append(1)
            else:
                placementData[3].append(0)

            # ax = dfAux.plot(kind='scatter', x='BallX', y='BallY', color='red')
            # placeCircle = plt.Circle((dfAux.loc[dfAux['PlaceX'].last_valid_index(), ['PlaceX','PlaceY']].values), 0.15, fill=False)
            # ax.add_artist(placeCircle)

            # for val,mark in zip([0,100,200],['x','*','+']):
            #     ax = plt.scatter(*(dfAux.loc[dfAux['BallX'].last_valid_index() - val, ['BallX','BallY']].values), marker=mark, s=100)

            # for val,mark in zip([0,100,200],['o','v','D']):
            #     plt.scatter(*(dfAux.loc[dfAux['BallX'].first_valid_index() + val, ['BallX','BallY']].values), marker=mark, s=100)
            # plt.show()

            dfTeamFiltered = dfTeamFiltered.append(dfAux)

    print(dfTeamFiltered.head())
    # dfTeamFiltered.plot(kind='scatter', x='BallX', y='BallY', color='blue')

    # plt.show()

    # placementTime, ballTraveledDist, ballPlacePointInitDist, validPlacement
    ballTravDistRatio = [ x/y for x,y in zip(placementData[1],placementData[2])]

    dfPlacements = pd.DataFrame({
                                "placementTime": placementData[0],
                                "ballTraveledDist": placementData[1],
                                "ballPlacePointInitDist": placementData[2],
                                "validPlacement": placementData[3],
                                "ballTravDistRatio" : ballTravDistRatio
                                })

    print(dfPlacements.head())

    # dfPlacements.loc[:, 'placementTime'].plot(kind='line')
    # dfPlacements.loc[:, 'validPlacement'].plot(kind='line')
    # plt.show()

    print("Correlations:\n", dfPlacements.corr(), '\n', dfPlacements.describe())
    print("Total placement time:", dfPlacements.loc[:, 'placementTime'].sum()/60, "min")

    dfPlacements.plot(kind='scatter', x='ballTravDistRatio', y='placementTime', s = dfPlacements['validPlacement']*100 + 10)
    plt.show()

if __name__ == "__main__":
    placeStats(sys.argv[1])