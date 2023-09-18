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


class StructureTo3DiBase:

    # Constantes
    CENTROID_CNT = 20
    INVALID_STATE = 2  # designa resíduos inválidos para o estado coil
    DISTANCE_ALPHA_BETA = 1.5336
    PI = 3.14159265359
    FEATURE_CNT = 10
    EMBEDDING_DIM = 2
    VIRTUAL_CENTER = {"alpha": 270, "beta": 0, "d": 2}

    centroids = [
        [-1.0729, -0.3600],
        [-0.1356, -1.8914],
        [0.4948, -0.4205],
        # ... (existem mais centróides que podem ser adicionados)
    ]

    # Operações vetoriais

    @staticmethod
    def add(a, b):
        return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]

    @staticmethod
    def sub(a, b):
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    @staticmethod
    def norm(a):
        length = (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5
        return [i / length for i in a]

    @staticmethod
    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

    @staticmethod
    def scale(a, f):
        return [i * f for i in a]

    @staticmethod
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
