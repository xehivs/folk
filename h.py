import numpy as np

def load_keel(string, separator=","):
    try:
        f = open(string, "r")
        s = [line for line in f]
        f.close()
    except:
        raise Exception

    s = filter(lambda e: e[0] != '@', s)
    s = [v.strip().split(separator) for v in s]
    df = np.array(s)
    X = np.asarray(df[:,:-1], dtype=float)
    d = {'positive': 1, 'negative': 0}
    y = np.asarray([d[v[-1].strip()] if v[-1].strip() in d else v[-1].strip() for v in s])

    return X, y
