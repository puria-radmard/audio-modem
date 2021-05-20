import numpy as np

def receive(soundchunk):
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
    
    def removeCP(signal):
        return signal[CP:]

    OFDM_RX_noCP = removeCP(soundchunk)

    def DFT(OFDM_RX):
        return np.fft.fft(OFDM_RX)
    OFDM_demod = DFT(OFDM_RX_noCP)

    OFDM_actual= OFDM_demod[1:64]
    
    Hest= 1

    def equalize(OFDM_actual, Hest):
        return OFDM_actual / Hest
    equalized_Hest = equalize(OFDM_actual, Hest)

    def get_payload(equalized):
        return equalized[dataCarriers]
    QAM_est = get_payload(equalized_Hest)

    def Demapping(QAM):
        # array of possible constellation points
        constellation = np.array([x for x in demapping_table.keys()])
        
        # calculate distance of each RX point to each possible point
        dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
        
        # for each element in QAM, choose the index in constellation 
        # that belongs to the nearest constellation point
        const_index = dists.argmin(axis=1)
        
        # get back the real constellation point
        hardDecision = constellation[const_index]
        
        # transform the constellation point into the bit groups
        return np.vstack([demapping_table[C] for C in hardDecision])
    
    PS_est = Demapping(QAM_est)

    def PS(bits):
        return bits.reshape((-1,))
    message = PS(PS_est)
    
    return(message)
