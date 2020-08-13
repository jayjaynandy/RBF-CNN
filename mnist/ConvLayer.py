import os
import numpy as np
import json
import Local
import EMSteps

def loadPretrained(config):
    config = json.load(open(config['layerDir'] + "config" + str(config['layerNum']) + ".json"))
    nodeMean = np.load(config['layerDir'] + "mean" + str(config['layerNum'])+".npy")
    nodeCov = np.load(config['layerDir'] + "conv" + str(config['layerNum'])+".npy")
    return config, nodeMean, nodeCov


def runner(config, data, batch_size, loadTrue= True, totalIter = 550, nonParamMode= 500):
    Local.memberThreshold = config['memThreshold']
    Local.alpha = config['alpha']
    if loadTrue:
        config, nodeMean, nodeCov = loadPretrained(config)
        Local.nodeCount = nodeMean.shape[0]
        Local.batchCount = updateBatchCount(config, Local.nodeCount+1)
        Local.alpha = 2.0
    else:
        nodeMean, nodeCov = EMSteps.initNodes(config, Local.nodeCount)        
    
    statMean, statDigCov, clusterCount = EMSteps.initStores(config, Local.nodeCount + 1)

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
            EMSteps.EStep(config, Local.updateMode, myBatch, nodeMean, nodeCov, 
                          statMean, statDigCov, clusterCount)  # updateStats
            batchSeen += 1
            iterCount += 1

            if batchSeen == Local.batchCount:  # update nodes
                np.set_printoptions(suppress=True)
                newMembership, nodeMean, nodeCov = EMSteps.MStep(config, clusterCount, statMean, statDigCov)
                Local.nodeCount = nodeMean.shape[0]
                Local.alpha = updareAlpha(Local.nodeCount, newMembership, Local.alpha)
                Local.batchCount = updateBatchCount(config, Local.nodeCount+1)

                statMean, statDigCov, clusterCount = EMSteps.initStores(config, Local.nodeCount + 1)
                batchSeen = 0
                updateCount += 1
 
                print ("batchCount "+ str(Local.batchCount)+ "epoch" + str(epochs)+ "creating store : " 
                       + str(Local.nodeCount + 1)+" Iter:" + str(updateCount))
                
                # save after every 50 steps
                if (updateCount %50== 0) and (updateCount>0):
                    print ("checkpoint saving ...")
                    config['outChannel']=nodeMean.shape[0]
                    writeConfig(config)
                    writeModel(config, nodeMean, nodeCov)
                    
                    ''' Visualize rbf means'''
                    Utils.imwrite(Utils.immerge(nodeMean, 
                                                np.floor(np.sqrt(nodeMean.shape[0])).astype(int)+1, 
                                                np.floor(np.sqrt(nodeMean.shape[0])).astype(int)+1), 
                                                'rbf_mean'+str(updateCount)+'.png')                
            if updateCount > nonParamMode:
                Local.updateMode = 2
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
    myF = float(newMembership) / float(Local.memberThreshold)
    myS = float(nodeCount) / float(Local.desiredNodeCount)
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
