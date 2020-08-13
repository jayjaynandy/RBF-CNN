import numpy as np
from numba import guvectorize, float64, int64
import Local


def EStep(config, updateMode, batch, nodeMean, nodeCov, 
          statMean, statDigCov, clusterCount):
    #initialize parameters
    fSize = config['filterSize']
    step = config['stride']
    batchSize = batch.shape[0]
    inRow = batch.shape[1]
    outChannel = int(nodeMean.shape[0])
    outRow = int((inRow-fSize)/step+1)

    outBatch = np.zeros((batchSize,inRow,inRow,outChannel)).astype(np.float64)
    outBatch = euclideanDist(batch, nodeMean, step, outBatch)
    outBatch = outBatch[:,0:outRow,0:outRow,0:outChannel]
    outBatchTmp = np.zeros((batchSize,outRow,outRow,outChannel)).astype(np.float64)
    outBatch = getLikelihood(nodeCov, outBatch, outBatchTmp)
    
    newBatch = np.zeros((batchSize,outRow,outRow,1)).astype(np.float64)
    outBatch = np.concatenate([outBatch, newBatch], axis=-1)
    outBatchTmp = np.zeros((batchSize, outRow, outRow, outChannel + 1)).astype(np.float64)

    if updateMode < 2:
        outBatch[:,:,:,-1] = -2.5
        outBatch = getProbMat(outBatch, outBatchTmp)
        outBatch[:,:,:,-1] = Local.alpha*outBatch[:,:,:,-1]
    if updateMode == 2:
        outBatch = getSigmoidLikelihood(outBatch, outBatchTmp)
        outBatch[:,:,:,-1] = 0
    
    updateStats(batch, step, updateMode, nodeMean, outBatch, statMean, statDigCov, clusterCount)

# dimensions
# batch  : (count, row, col, channel)
# filter : (filtercount, fsize, fsize, channel(d))
# outBatch : (count, outRow, outCol, channel(f))

@guvectorize([(float64[:,:,:,:], float64[:,:,:,:], int64[:], float64[:,:,:,:])], 
              '(d1,d2, d2, d3),(f1, f2, f2, d3), ()->(d1, d2, d2, f1)', target='parallel')
def euclideanDist(batch, nodeMean, step, outBatch):    
    for b in range(0, batch.shape[0]):                                          # each image
        for f in range(0, nodeMean.shape[0]):                                   # each filter
            for r in range(0, batch.shape[1]-nodeMean.shape[1]+1, step[0]):     # starting row of img
                for c in range(0, batch.shape[2]-nodeMean.shape[2]+1, step[0]): # starting col of img
                    outR = int(r/step[0])
                    outC = int(c/step[0])
                    for w in range(0,nodeMean.shape[1]):                        # iterate over filter/img row
                        for h in range(0,nodeMean.shape[2]):                    # iterate over filter/img col
                            for d in range(0,batch.shape[3]):                   # iterate over filter/img channel
                                outBatch[b,outR, outC, f] += (batch[b,r+w,c+h,d]-nodeMean[f,w,h,d])**2


# outBatch : (count, outRow, outCol, channel(d))
@guvectorize([(float64[:], float64[:,:,:,:], float64[:,:,:,:])], 
              '(d3), (d1,d2, d2, d3)->(d1, d2, d2, d3)', target='parallel')
def getLikelihood(nodeCov, outBatch, outBatchTmp):
    for b in range(0, outBatch.shape[0]):
        for r in range(0, outBatch.shape[1]):
            for c in range(0, outBatch.shape[2]):
                for d in range(0,outBatch.shape[3]):
                    outBatchTmp[b,r,c,d] = -outBatch[b,r,c,d]/(2*nodeCov[d])-np.log(nodeCov[d])*0.5

#%%
# outBatch : (count, outRow, outCol, channel(d))
@guvectorize([(float64[:,:,:,:], float64[:,:,:,:])], 
              '(d1,d2, d2, d3)->(d1, d2, d2, d3)', target='parallel')
def getSigmoidLikelihood(outBatch, outBatchTmp):
    for b in range(0, outBatch.shape[0]):
        for r in range(0, outBatch.shape[1]):
            for c in range(0, outBatch.shape[2]):
                for d in range(0,outBatch.shape[3]):
                    outBatchTmp[b,r,c,d] = 1/(1+np.exp(-outBatch[b,r,c,d]))

@guvectorize([(float64[:,:,:,:], float64[:,:,:,:])], 
              '(d1,d2, d2, d3)->(d1, d2, d2, d3)', target='parallel')
def getProbMat(outBatch, outBatchTmp):
    for b in range(0, outBatch.shape[0]):
        for r in range(0, outBatch.shape[1]):
            for c in range(0, outBatch.shape[2]):
                threshold = 50-np.max(outBatch[b,r,c])
                outBatchTmp[b,r,c] = np.exp(outBatch[b,r,c]+threshold)



@guvectorize([(float64[:,:,:,:], int64[:],int64[:],float64[:,:,:,:], float64[:,:,:,:], 
               float64[:,:,:,:], float64[:,:,:,:], float64[:])], 
              '(d1,d2, d2, d3),(),(),(f1,f2,f2,d3),(d1,o1,o2,o3)->(o3,f2,f2,d3),(o3,f2,f2,d3),(o3)', target='parallel')
    
def updateStats(batch, step, updateMode, nodeMean, outBatch, statMean, statDigCov, clusterCount):
    for b in range(0, batch.shape[0]):                                          # each image
        for r in range(0, batch.shape[1]-nodeMean.shape[1]+1, step[0]):            # starting row of img
            for c in range(0, batch.shape[2]-nodeMean.shape[1]+1, step[0]):        # starting col of img
                outR = int(r/step[0])
                outC = int(c/step[0])
                f = np.argmax(outBatch[b,outR,outC,:])
                clusterCount[f] +=1
                
                for w in range(0,nodeMean.shape[1]):                            # iterate over filter/img row
                    for h in range(0,nodeMean.shape[1]):                        # iterate over filter/img col
                        for d in range(0,batch.shape[3]):                       # iterate over filter/img channel
                            statMean[f,w,h,d] += batch[b,r+w,c+h,d]
                            statDigCov[f,w,h,d] += batch[b,r+w,c+h,d]**2

#%%

def initStores(config, count):
    #initialize parameters
    fSize = config['filterSize']
    inChannel = config['inChannel']
    
    statMean = np.zeros((count,fSize,fSize,inChannel)).astype(np.float64)
    statCov = np.zeros((count,fSize,fSize,inChannel)).astype(np.float64)
    clusterCount = np.zeros((count)).astype(np.float64)
    
    return statMean, statCov, clusterCount
    

def MStep(config, clusterCount, statMean, statDigCov):
    #initialize
    fSize = config['filterSize']
    fCount = statMean.shape[0]
    fChannel = config['inChannel']
    
    nodeMean = np.zeros((fCount,fSize,fSize,fChannel)).astype(np.float64)
    nodeCov = np.zeros((fCount)).astype(np.float64)
    countSum, countNode = updateParam(statMean, statDigCov, clusterCount, nodeMean, nodeCov)
    
    nodeMean = nodeMean[:countNode]
    nodeCov = nodeCov[:countNode]
    newMembership = clusterCount[-1]/countSum
    
    if np.isnan(nodeMean).any():
        newMembership =-1
    if np.isnan(nodeCov).any():
        newMembership =-1
        
    return (newMembership, nodeMean, nodeCov)



def updateParam(statMean, statDigCov, clusterCount, 
                nodeMean, nodeCov):
    countSum = np.sum(clusterCount)
    countNode = 0
    for f in range(0, nodeMean.shape[0]):
        if clusterCount[f] > Local.countThreshold:
            if (f<nodeMean.shape[0]-1) or (f==nodeMean.shape[0]-1 and clusterCount[f]/countSum > Local.memberThreshold):
                    nodeMean[countNode] = statMean[f]/clusterCount[f]
                    nodeCov[countNode] = np.sum(statDigCov[f] / clusterCount[f]) - np.sum(nodeMean[countNode] ** 2) + 1E-20
                    if nodeCov[countNode] < 1E-20:
                        nodeCov[countNode] = 1E-20
                    countNode +=1

    return (countSum, countNode)
                
def initNodes(config, count):
    #initialize parameters
    fSize = config['filterSize']
    inChannel = config['inChannel']
    
    nodeMean = np.random.rand(count,fSize,fSize,inChannel).astype(np.float64)
    covVal = (fSize**2)*inChannel
    nodeCov = covVal*np.ones(count).astype(np.float64)
    
    return (nodeMean, nodeCov)    
    
