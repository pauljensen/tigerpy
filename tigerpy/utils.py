
def mapl(f, lst):
    return list(map(f, lst))
    
def flatten(lst):
    return [item for sublist in lst for item in sublist]

def unique(lst):
    return list(set(lst))