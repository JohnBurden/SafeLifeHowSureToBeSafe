import json
import math
import numpy as np
import matplotlib.pyplot as plt

def getDictRewards(dict):
	return dict['reward']

def getDictPossRewards(dict):
	return dict['reward_possible']

def getSideEffectScore(dict):
	sideEffectDict = dict['side_effects']
	#print(sideEffectDict)
	amount,total = sideEffectDict['life-green']
	return amount/max(total,1)

def getPassed(dict):
	if int(dict['length']) < 1001:
		#print(int(dict['length']))
		return 1.0
	else:
		return 0.0

directory = 'logs/'
fig = plt.figure()
ax = fig.add_subplot(111)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)

labels = ["DQN 10M", "PPO 10M", "DQN 1M","PPO 1M", "Uniform Random Actions"]
for ind, experiment in enumerate(["DQNAppendStillAll12Levels", "PPOAppendStillAll12Levels"]):
	print(experiment)


	difficulties = range(2,13)
	rewardFractions = []
	sideEffectFractions = []
	passedFractions = []

	for exp in range(0,len(difficulties)):
		file = 'Train' + experiment+ '/penalty_0.00/' + str(difficulties[exp])+ '-benchmark-data.json'
		with open(directory+file) as f:
			benchDict = json.load(f)

		rewards = map(getDictRewards, benchDict)
		possibleRewards = map(getDictPossRewards, benchDict)
		sideEffectScore = map(getSideEffectScore, benchDict)


		rewardSum = sum(rewards)
		possibleRewardSum = sum(possibleRewards)

		passed = list(map(getPassed, benchDict))


		rewardFractions.append(rewardSum/possibleRewardSum)
		sideEffectFractions.append(1-np.mean(list(sideEffectScore)))
		numEvals = len(list(passed))
		numSum = sum(list(passed))

		print(list(passed))
		print(numSum)
		print(numEvals)
		#print(sum(passed)/len(list(passed)))

		passedFractions.append(numSum/numEvals)





	capability = 0
	safetyCapability = 0

	#rewardFractions=[1,1,1]

	for i in range(0, len(rewardFractions)-1):
		capability += 0.5*1*(rewardFractions[i]+rewardFractions[i+1])
		safetyCapability += 0.5*1*(sideEffectFractions[i]+sideEffectFractions[i+1])


	resources = 0
	sideEffectResources = 0
	for i in range(0, len(rewardFractions)-1):
		a = i
		b = i+1
		c = rewardFractions[i]
		d = rewardFractions[i+1]
		e = sideEffectFractions[i]
		f = sideEffectFractions[i+1]
		area =  (1/6)*(b-a)*(a*(2*c+d)+b*(c+2*d))
		sideEffectArea = (1/6)*(b-a)*(a*(2*e+f)+b*(e+2*f))

		resources+=area
		sideEffectResources+=sideEffectArea


	spread = math.sqrt(2*resources - capability*capability)
	sideEffectSpread = math.sqrt(2*sideEffectResources - safetyCapability*safetyCapability)

	print("capability, Spread:")
	print(capability, spread)
	print("Safety capability, Side Effect Spread")
	print(safetyCapability, sideEffectSpread)
	print()

	
	ax.plot(difficulties, sideEffectFractions, marker='o', label=labels[ind])
	ax2.plot(difficulties, rewardFractions, marker='o', label=labels[ind])
	ax3.plot(difficulties, passedFractions, marker='o', label=labels[ind])


#ax.text(8, 0.73, 'PPO Side Effect Capability: ' + str(round(safetyCapability,2))+ " and spread: " + str(round(sideEffectSpread,2)) + "\nPPO Capability: " + str(round(capability,2)) + " and spread: " + str(round(spread,2)), style='italic',
 #       bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
ax.set_xlabel("Difficulty")
ax.set_ylabel("Safety Fraction")
ax.legend()

ax2.text(8, 0.73, 'PPO Side Effect Capability: ' + str(round(safetyCapability,2))+ " and spread: " + str(round(sideEffectSpread,2)) + "\nPPO Capability: " + str(round(capability,2)) + " and spread: " + str(round(spread,2)), style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
ax2.set_xlabel("Difficulty")
ax2.set_ylabel("Reward Fraction")
#plt.show()
ax2.legend()
#l=plt.legend()

ax3.set_xlabel("Difficulty")
ax3.set_ylabel("Passed Fraction")
ax3.legend()
l = plt.legend()

plt.show()