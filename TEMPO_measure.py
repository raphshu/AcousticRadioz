import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


#This function get as input the bpm and onset of two songs and return the distance between them in manners of beat.
#the smaller the value - the better
def TEMPO_measure(x,y,alpha=1):
    # logging.debug('Start TEMPO-measure')
    dist_bpm = []
    dist_or = []
    for i in [1,2,4]:
        j = i-1
        dist_bpm.append(pow(alpha,j) * abs((max(x[0],y[0])/min(x[0],y[0]))-i))
        dist_or.append(pow(alpha,j) * abs((max(x[1],y[1])/min(x[1],y[1]))-i))

    # logging.debug('Finish TEMPO-measure')
    return (min(dist_bpm)+min(dist_or))/2

