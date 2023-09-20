import math

class Alphabet3diSeqDist:
    CENTROID_CNT = 20
    INVALID_STATE = CENTROID_CNT
    centroids = [-284,-147,-83,-52,-33,-21,-13,-7,-4,-3,-1,1,3,7,13,24,40,68,123,250]


class StructureTo3diSeqDist:
    
    def __init__(self):
        self.states = []
        self.partnerIdx = []
        self.mask = []

    @staticmethod
    def createResidueMask(mask, ca, n, c, length):
        for i in range(length):
            mask[i] = not (math.isnan(ca[i][0]) or math.isnan(n[i][0]) or math.isnan(c[i][0]))

    @staticmethod
    def findResiduePartners(partnerIdx, cb, validMask, n):
        # Escolhe para cada resíduo o vizinho mais próximo como parceiro
        # em termos de distâncias entre seus centros virtuais/C_betas.
        # Ignora os primeiros/últimos e resíduos inválidos.
        for i in range(1, n - 1):
            minDistance = float('inf')
            for j in range(1, n - 1):
                if i != j and validMask[j]:
                    dist = StructureTo3Di.calcDistanceBetween(cb[i], cb[j])
                    if dist < minDistance:
                        minDistance = dist
                        partnerIdx[i] = j
            if partnerIdx[i] == -1:  # nenhum parceiro encontrado
                validMask[i] = False
            
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
        """
            Esta função parece percorrer cada posição do vetor de entrada, calcular a distância sequencial e então encontrar o centroide mais próximo dessa distância. 
            Em seguida, ele atribui o estado (que é o índice do centroide mais próximo) à posição atual do vetor de saída states.
        """
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


# Esta classe StructureTo3DiBase serve como a base para as operações relacionadas à estrutura 3Di. 
