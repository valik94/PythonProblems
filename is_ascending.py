def is_ascending(items):
    #first 2 edge cases if items=[] and len(items)==1
    if len(items)<2:
        return True
    flag=0
    i=1
    while i<=len(items):
        if (items[1]<items[i-1]):
            flag=0
        i+=1
    if (not flag):
        return True
    elif (flag):
        return False 
"""
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            if(items[i]<items[j]):
                return True
            if (items[i]<=items[j]):
                return False
"""

print(is_ascending([0,1,2,3,4,6,5]))