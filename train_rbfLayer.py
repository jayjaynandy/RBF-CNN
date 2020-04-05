import ConvLayer as gcn
import numpy as np
import Utils





'''  layer configuration '''
layerNum = 0
config = {'layerDir':'./rbf_layer/',
          'alpha':1.0,
          'memThreshold':0.001}

layerNum = 0
config['layerNum'] = layerNum
config['filterSize'] = 3
config['stride'] = 1



if __name__ == '__main__':
    data, _ = Utils.npzLoad('./extras/mnist.npz')
    if np.max(data) >1:
        data = data/255
        
    config['inChannel'] = int(data.shape[3])
    config['inSize'] = int(data.shape[1])
    config['outSize'] = (config['inSize']-config['filterSize'])/config['stride'] +1
    nodeMean, nodeCov = gcn.runner(config= config, data= data, 
                                   batch_size=50, loadTrue=False,
                                   totalIter = 500, nonParamMode= 250)
    
    ''' Visualize rbf means'''
    Utils.imwrite(Utils.immerge(nodeMean, 
                                np.floor(np.sqrt(nodeMean.shape[0])).astype(int)+1, 
                                np.floor(np.sqrt(nodeMean.shape[0])).astype(int)+1), 
                                'mnistFashion3x3.png')
