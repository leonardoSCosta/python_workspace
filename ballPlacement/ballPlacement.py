import sys
import os
import struct 
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.spatial import distance
import pandas as pd

import referee_pb2 as referee
import messages_robocup_ssl_detection_pb2 as detection
import messages_robocup_ssl_geometry_pb2 as geometry
import messages_robocup_ssl_refbox_log_pb2 as refbox
import messages_robocup_ssl_wrapper_pb2 as wrapper

FILE_FORMAT_SIZE = 16 # 16 bytes = int64 + 2 x int32

MESSAGE_TYPE = {
	0 : "Blank Message",
	1 : "Unknown",
 	2 : "SSL Vision 2010",
	3 : "SSL Refbox 2013",
	4 : "SSL Vision 2014"
	}

def getHeader(f):
	return f.read(12)

def getVersion(f):
	return int.from_bytes(f.read(4), byteorder = "big") 

def getDecodedMessage(binaryLine):
	# >qii -> format for int64 + int32 + int32 in big-endian
	data = struct.unpack('>qii',binaryLine) 
	timestamp = datetime.timedelta(seconds=data[0]/1e9)

	messageType = MESSAGE_TYPE.get(data[1])
	messageSize = data[2]
	#print("Message Size =", messageSize, "Message Type =", messageType, 
	#	"Timestamp =", timestamp.split(', ')[-1])

	return timestamp, messageType, messageSize

def getMessageFormat(f):
	return f.read(FILE_FORMAT_SIZE)

def checkFile(f):
	header = getHeader(f)
	version = getVersion(f)
	fileOK = False

	if(header == b'SSL_LOG_FILE'):
		print("File Header",header)
		fileOK = True
	else:
		print("Wrong file header. Found header =", header)
		fileOK = False

	if(version == 1):
		print("File Version =",version)
		fileOK = True & fileOK
	else:
		print("Wrong file version. Found version =", version)
		fileOK = False & fileOK

	return fileOK

def getBallPosition(frame, prevX, prevY):
	if(len(frame.detection.balls) > 0 ):
		#print("Ball(",frame.detection.balls[0].x,",",frame.detection.balls[0].y,")")
		return frame.detection.balls[0].x, frame.detection.balls[0].y
	return prevX,prevY

def getRefereeCommand(frame):
	return frame.command

def getBallPlacementPosition(frame):
	return frame.designated_position.x, frame.designated_position.y

def getTeamName(frame):
	if frame.command in [referee.SSL_Referee.PREPARE_KICKOFF_YELLOW, 
			     referee.SSL_Referee.PREPARE_PENALTY_YELLOW, 
			     referee.SSL_Referee.DIRECT_FREE_YELLOW, 
			     referee.SSL_Referee.INDIRECT_FREE_YELLOW, 
			     referee.SSL_Referee.TIMEOUT_YELLOW, 
			     referee.SSL_Referee.BALL_PLACEMENT_YELLOW]:
		return frame.yellow.name
	elif frame.command in [referee.SSL_Referee.PREPARE_KICKOFF_BLUE, 
			       referee.SSL_Referee.PREPARE_PENALTY_BLUE, 
			       referee.SSL_Referee.DIRECT_FREE_BLUE, 
			       referee.SSL_Referee.INDIRECT_FREE_BLUE, 
			       referee.SSL_Referee.TIMEOUT_BLUE, 
			       referee.SSL_Referee.BALL_PLACEMENT_BLUE]:
		return frame.blue.name
	return ""

def saveBallTrajectory(ballX, ballY, placeX, placeY, figTitle):
	# clear data
	ax = plt.gca()
	del ax.collections[:]

	plt.title("Ball position")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.grid(True)
	plt.ylim(-5,5)
	plt.xlim(-7,7)

	plt.scatter(placeX, placeY, s=250, c='green')

	if(distance.euclidean([ballX[-1],  ballY[-1]  ], 
   						  [placeX[-1], placeY[-1] ] )  < 0.15):
		plt.scatter(ballX[:-1], ballY[:-1], s=15, c='orange', marker='x')
	else:
		plt.scatter(ballX[:-1], ballY[:-1], s=15, c='red', marker='x')

	plt.scatter(ballX[-1], ballY[-1], s=100, c='blue', marker='*')

	placeX.clear()
	placeY.clear()
	ballX.clear()
	ballY.clear()
	plt.savefig(figTitle, dpi=300, bbox_inches='tight')
	#plt.show()
	#plt.pause(0.001)
	#plt.close()

def readSSL_Log():
	codeTime = time.time()
	try:
		path = sys.argv[1]
		logFile = open(path,'rb')
		print("\n", logFile, end="\n\n")
	except:
		print("Couldn't open file", end="\n\n")
		return

	totalBytes = os.path.getsize(path)
	bytesRead = 0

	wrapperFrame = wrapper.SSL_WrapperPacket()
	myRefereeFrame = referee.SSL_Referee()

	if(checkFile(logFile) == True):
		line = getMessageFormat(logFile)
		bytesRead += 32

		bX, bY = 0, 0
		ballPos = [[],[],[],[]]# X,Y for team A, X,y for team B
		placePos = [[],[],[],[]]# same as ballPos
		timestampList = [[],[]]
		placementTime = [[],[]]# blue placement time, yellow placement time
		teams = []
		teamDict = {}
		input("Press Enter to continue")

		currentRefCommand = referee.SSL_Referee.HALT
		commandChanged = False
		initialTimestamp = datetime.timedelta(seconds=0)
		remainingPercent = 100
		teamName = ""

		while line:
			timeStamp, messageType, messageSize = getDecodedMessage(line)
			messageData = logFile.read(messageSize)

			if messageType == "SSL Refbox 2013":
				myRefereeFrame.ParseFromString(messageData)

				if teams == []:
					teams = [myRefereeFrame.blue.name, myRefereeFrame.yellow.name]
					teamDict = {teams[0]: 0, teams[1]: 2}

				currentRefCommand = getRefereeCommand(myRefereeFrame)

			elif messageType == "SSL Vision 2014":
				wrapperFrame.ParseFromString(messageData)
                                bX, bY = getBallPosition(wrapperFrame, bX, bY)

			if(currentRefCommand == referee.SSL_Referee.BALL_PLACEMENT_YELLOW or
				currentRefCommand == referee.SSL_Referee.BALL_PLACEMENT_BLUE):
				placeX, placeY = getBallPlacementPosition(myRefereeFrame)
				teamName = getTeamName(myRefereeFrame)

				if teamName in teamDict:
					ballPos[teamDict[teamName]].append(bX/1e3)
					ballPos[teamDict[teamName]+1].append(bY/1e3)
					placePos[teamDict[teamName]].append(placeX/1e3)
					placePos[teamDict[teamName]+1].append(placeY/1e3)

					if teamName in teams:
						timestampList[teams.index(teamName)].append(timeStamp.total_seconds())
				else:
					print("Team not found:", teamName, teamDict)
					# input()
			
				if(commandChanged == False):
					commandChanged = True
					initialTimestamp = timeStamp
			else:
				if(commandChanged == True):
					deltaT = (timeStamp - initialTimestamp)
					deltaT = deltaT.seconds + 1e-6*deltaT.microseconds

					if teamName in teams:
						placementTime[teams.index(teamName)].append(deltaT)

					# saveBallTrajectory(ballPos[teamDict[teamName]], ballPos[teamDict[teamName]+1], 
									#    placePos[teamDict[teamName]], placePos[teamDict[teamName]+1], 
										# 'Placements/BallPlacement'+ teamName +'_'+str(deltaT)+'.png')
					#print("Ball placement took ", timeStamp-initialTimestamp)
					commandChanged = False

			line = getMessageFormat(logFile)
			bytesRead += 16 + messageSize

			auxPercent = round(10000*(totalBytes-bytesRead)/totalBytes)/100 
			if(auxPercent < remainingPercent):
				remainingPercent = auxPercent
				print("\n", remainingPercent,"% Remaining")

		print("Finished reading log file.", end=" ")
		print("Took ",round(100*time.time() - 100*codeTime)/100, "s")

		dataColumn = ['Timestamp', 'PlaceX', 'PlaceY', 'BallX', 'BallY']
		dfPlace = [pd.DataFrame(), pd.DataFrame()]

		dataDict = { "Timestamp": timestampList,
			     "PlaceX"   : [l for index,l in enumerate(placePos)
                                                            if index in [0,2]],
			     "PlaceY"   : [l for index,l in enumerate(placePos)
                                                            if index in [1,3]],
			     "BallX"    : [l for index,l in enumerate(ballPos)
                                                            if index in [0,2]],
			     "BallY"    : [l for index,l in enumerate(ballPos)
                                                            if index in [1,3]]}

		for i in range(0, len(teams)):
			for col in dataColumn:
				if col in dataDict:
					data = dataDict[col]
					dfPlace[i][col] = data[i]
				else:
					print("Column not found:", col)
					input()

			dfPlace[i].rename_axis(teams[i], inplace=True)
			print(dfPlace[i].head())
			dfPlace[i].to_csv(teams[i]+"_BallPlacement.csv")
			dfPlace[i].plot(kind='scatter', x= 'BallX', y='BallY')
			plt.show()

		dfPlace[0] = dfPlace[0].append(dfPlace[1])	

		dfPlace[0].to_csv(teams[0] + teams[1] + "_BallPlacement.csv")
		

	logFile.close()

if __name__ == "__main__":
	readSSL_Log()
