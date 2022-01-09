import random
import copy

def Generate(n,m):
    PT = []
    MT = []
    for i in range(n):
        PT_i = [random.randint(5,40) for i in range(m)]
        MT_i = [_ for _ in range(m)]
        random.shuffle(MT_i)
        PT.append(PT_i)
        MT.append(copy.copy(MT_i))
    return PT,MT

