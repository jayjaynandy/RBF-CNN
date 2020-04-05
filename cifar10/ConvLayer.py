import os
import numpy as np
import json

import Locals
import EMSteps

def loadPretrained(config):
    
    config = json.load(open(config['layerDir'] + "config" + str(config['layerNum']) + ".json"))
    nodeMean = np.load(config['layerDir'] + "mean" + str(config['layerNum'])+".npy")
    nodeCov = np.load(config['layerDir'] + "conv" + str(config['layerNum'])+".npy")
    return config, nodeMean, nodeCov

def runner(config, data, batch_size, loadTrue= True, totalIter = 550, nonParamMode= 500):
    # initialize node store
    Locals.memberThreshold = config['memThreshold']
    Locals.alpha = config['alpha']
    if loadTrue:
        config, nodeMean, nodeCov = loadPretrained(config)
        Locals.nodeCount = nodeMean.shape[0]
        Locals.batchCount = updateBatchCount(config, Locals.nodeCount+1)
        Locals.alpha = 2.0
    else:
        nodeMean, nodeCov = EMSteps.initNodes(config, Locals.nodeCount)        
    
    statMean, statDigCov, clusterCount = EMSteps.initStores(config, Locals.nodeCount + 1)

    # iterate
    batchSeen = 0
    iterCount = 0
    updateCount = 0
    epochs = 0
    data_count = int(data.shape[0])

    while 1:
        epochs +=1
        for j in range(int(data_count/batch_size)):            
            
            myBatch = data[j*batch_size: (j+1)*batch_size]
            EMSteps.EStep(config, Locals.updateMode, myBatch, nodeMean, nodeCov, statMean, statDigCov, clusterCount)  # updateStats
            batchSeen += 1
            iterCount += 1

            if batchSeen == Locals.batchCount:  # update nodes
                np.set_printoptions(suppress=True)
                newMembership, nodeMean, nodeCov = EMSteps.MStep(config, clusterCount, statMean, statDigCov)
                Locals.nodeCount = nodeMean.shape[0]
                Locals.alpha = updareAlpha(Locals.nodeCount, newMembership, Locals.alpha)
                Locals.batchCount = updateBatchCount(config, Locals.nodeCount+1)

                statMean, statDigCov, clusterCount = EMSteps.initStores(config, Locals.nodeCount + 1)
                batchSeen = 0
                updateCount += 1
 
                print ("batchCount "+ str(Locals.batchCount)+ "epoch" + str(epochs)+ "creating store : " 
                       + str(Locals.nodeCount + 1)+" Iter:" + str(updateCount))
                
                # save after every 50 steps
                if (updateCount %50== 0) and (updateCount>0):
                    print ("checkpoint saving ...")
                    config['outChannel']=nodeMean.shape[0]
                    writeConfig(config)
                    writeModel(config, nodeMean, nodeCov)
                
            if updateCount > nonParamMode:
                Locals.updateMode = 2
            if updateCount > totalIter:
                break
            
        if updateCount > totalIter:
            break        
        
    config['outChannel']=nodeMean.shape[0]
    writeConfig(config)
    writeModel(config, nodeMean, nodeCov)
    return (nodeMean, nodeCov)

def updateBatchCount(config, nodeCount):
    batchSize = 64
    outSize = config['outSize']
    dataCount = batchSize * outSize*outSize
    reqdCount = nodeCount * 300
    
    batchCount = reqdCount/ dataCount
    if batchCount < 15:
        batchCount = 15
    if batchCount > 600:
        batchCount = 600
    return batchCount

def updareAlpha(nodeCount, newMembership, alpha):
    myF = float(newMembership) / float(Locals.memberThreshold)
    myS = float(nodeCount) / float(Locals.desiredNodeCount)
    # print myS
    alpha = alpha + (0.6913 / ((1 + np.exp(myS)) * (1 + np.exp(myF))) - 0.05)
    print (":node Count : " + str(nodeCount) + " memVal : " + str(newMembership) + " alpha : " + str(alpha))
    return alpha


def writeConfig(config):
    if not os.path.exists(config['layerDir']):
        os.makedirs(config['layerDir'])

    fName = config['layerDir'] + "config" + str(config['layerNum']) + ".json"
    print (fName)
    with open(fName, 'w') as fp:
        json.dump(config, fp)

def writeModel(config, nodeMean, nodeCov):
    np.save(config['layerDir'] + "mean" + str(config['layerNum']), nodeMean)
    np.save(config['layerDir'] + "conv" + str(config['layerNum']), nodeCov)