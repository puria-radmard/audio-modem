import numpy as np

def emmit(bitschunk):

    K = 64
    CP = K//4  
    P = 8 
    pilotValue = 3+3j 

    allCarriers = np.arange(K) 

    pilotCarriers = allCarriers[::K//P] 


    pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
    P = P+1


    dataCarriers = np.delete(allCarriers, pilotCarriers)

    mu = 4 
    payloadBits_per_OFDM = len(dataCarriers)*mu 

    mapping_table = {
        (0,0,0,0) : -3-3j,
        (0,0,0,1) : -3-1j,
        (0,0,1,0) : -3+3j,
        (0,0,1,1) : -3+1j,
        (0,1,0,0) : -1-3j,
        (0,1,0,1) : -1-1j,
        (0,1,1,0) : -1+3j,
        (0,1,1,1) : -1+1j,
        (1,0,0,0) :  3-3j,
        (1,0,0,1) :  3-1j,
        (1,0,1,0) :  3+3j,
        (1,0,1,1) :  3+1j,
        (1,1,0,0) :  1-3j,
        (1,1,0,1) :  1-1j,
        (1,1,1,0) :  1+3j,
        (1,1,1,1) :  1+1j
    }

    demapping_table = {v : k for k, v in mapping_table.items()}

    #bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, )) #insert bit sequence here
    #print(len(bits))

    bits= bitschunk

    def SP(bits):
        return bits.reshape((len(dataCarriers), mu))
    bits_SP = SP(bits)

    def Mapping(bits):
        return np.array([mapping_table[tuple(b)] for b in bits])
    QAM = Mapping(bits_SP)

    def OFDM_symbol(QAM_payload):
        symbol = np.zeros(K, dtype=complex)
        symbol[pilotCarriers] = pilotValue  
        symbol[dataCarriers] = QAM_payload 
        return symbol
    OFDM_data = OFDM_symbol(QAM)
    conju= np.conj(OFDM_data)
    reversed= conju[::-1]
    z=[0]
    OFDM_data_conjugated= np.concatenate((z,OFDM_data, reversed))
    # print(OFDM_data_conjugated)
    # print(len(OFDM_data_conjugated))
    def IDFT(x):
        return np.fft.ifft(x)
    OFDM_time = IDFT(OFDM_data_conjugated)


    def addCP(OFDM_time):
        cp = OFDM_time[-CP:]               
        return np.hstack([cp, OFDM_time])  
    OFDM_withCP = addCP(OFDM_time)
    message_sent= np.real(OFDM_withCP)

    return message_sent


