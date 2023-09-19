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



class StructureTo3Di(StructureTo3DiBase):


    @staticmethod
    def replaceCBWithVirtualCenter(ca, n, c, cb, length):
        for i in range(length):
            if cb[i] == [0, 0, 0]:
                # Compute the direction from CA to CB
                ca_to_cb = StructureTo3DiBase.sub(n[i], ca[i])
                ca_to_cb = StructureTo3DiBase.norm(ca_to_cb)

                # Scale the direction
                ca_to_cb = StructureTo3DiBase.scale(ca_to_cb, StructureTo3DiBase.DISTANCE_ALPHA_BETA)

                # Compute the virtual CB position
                virtual_cb = StructureTo3DiBase.add(ca[i], ca_to_cb)

                # Set the virtual CB position
                cb[i] = virtual_cb

    @staticmethod
    def degreeToRadians(degree):
        return degree * StructureTo3DiBase.PI / 180.0

    @staticmethod
    def calcVirtualCenter(virtual_center, c_alpha, c_beta, alpha, beta, d):
        """A função calcVirtualCenter usa a fórmula de rotação de Rodrigues para calcular o centro virtual entre os átomos CA e CB."""
        alpha = StructureTo3Di.degreeToRadians(alpha)
        beta = StructureTo3Di.degreeToRadians(beta)
        
        v = StructureTo3DiBase.sub(c_beta, c_alpha)
        
        # ângulo normal (entre CA-N e CA-VIRT)
        a = StructureTo3DiBase.sub(c_beta, c_alpha)
        b = StructureTo3DiBase.sub(n, c_alpha)
        k = StructureTo3DiBase.norm(StructureTo3DiBase.cross(a, b))  # eixo de rotação
        
        v = StructureTo3DiBase.add(StructureTo3DiBase.add(StructureTo3DiBase.scale(v, cos(alpha)),
                                StructureTo3DiBase.scale(StructureTo3DiBase.cross(k, v), sin(alpha))),
                                StructureTo3DiBase.scale(StructureTo3DiBase.scale(k, StructureTo3DiBase.dot(k, v)), 1 - cos(alpha)))

        # ângulo diedro (eixo: CA-N, CO, VIRT)
        k = StructureTo3DiBase.norm(StructureTo3DiBase.sub(n, c_alpha))
        v = StructureTo3DiBase.add(StructureTo3DiBase.add(StructureTo3DiBase.scale(v, cos(beta)),
                                StructureTo3DiBase.scale(StructureTo3DiBase.cross(k, v), sin(beta))),
                                StructureTo3DiBase.scale(StructureTo3DiBase.scale(k, StructureTo3DiBase.dot(k, v)), 1 - cos(beta)))
        
        virtual_center = StructureTo3DiBase.add(c_alpha, StructureTo3DiBase.scale(v, d))
        return virtual_center

    @staticmethod
    def calcDistanceBetween(a, b):
        """Para calcular a distância entre dois pontos em um espaço tridimensional."""
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return (dx**2 + dy**2 + dz**2)**0.5
