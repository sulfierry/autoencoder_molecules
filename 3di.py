class Alphabet3diSeqDist:
    CENTROID_CNT = 20
    INVALID_STATE = CENTROID_CNT
    centroids = [-284,-147,-83,-52,-33,-21,-13,-7,-4,-3,-1,1,3,7,13,24,40,68,123,250]

class StructureTo3diSeqDist:
    
    def __init__(self):
        self.states = []
        self.partnerIdx = []
        self.mask = []
            
    def structure2states(self, ca, n, c, cb, length):
        self.states.clear()
        self.partnerIdx.clear()
        self.mask.clear()

        if length > len(self.states):
            self.states = [-1] * length
            self.partnerIdx = [-1] * length
            self.mask = [False] * length

        # Estas funções precisam ser implementadas com base em suas implementações C++
        self.replaceCBWithVirtualCenter(ca, n, c, cb, length)
        self.createResidueMask(self.mask, ca, n, c, length)
        self.findResiduePartners(self.partnerIdx, cb, self.mask, length)
        self.discretizeSeqDistance(self.states, self.partnerIdx, self.mask, length)

        return self.states

    
    def discretizeSeqDistance(self, states, partnerIdx, mask, length):
        minDistance = float('inf')
        closestState = Alphabet3diSeqDist.INVALID_STATE

        for i in range(length):
            if mask[i]:
                minDistance = float('inf')
                seqDistance = partnerIdx[i] - i
                for j in range(Alphabet3diSeqDist.CENTROID_CNT):
                    distToCentroid = abs(Alphabet3diSeqDist.centroids[j] - seqDistance)
                    if distToCentroid < minDistance:
                        closestState = j
                        minDistance = distToCentroid
            states[i] = closestState
