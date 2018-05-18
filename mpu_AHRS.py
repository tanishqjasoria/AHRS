import time
import numpy

nowTime=time.time()
deltat = (nowTime - lastUpdate) #set integration time by time elapsed since last filter update
lastTime = nowTime