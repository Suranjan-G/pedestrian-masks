import pickle

def dumper(filename,objname):
    """
    dumper(filename,objname)
    
    Parameters:
        filename --> name of the file to dump
        objname  --> object to dump
    """
    with open(filename,"wb") as f:
        pickle.dump(objname,f)
    
def undumper(filename):
    """
    undumper(filename) --> object
    
    Parameters:
        filename --> name of the file to undump
        
    Return:
        object --> object to be undumped
    """
    with open(filename,"rb") as f:
        obj=pickle.load(f)
    return obj

