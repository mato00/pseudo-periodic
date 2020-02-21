import numpy as np

def CPSC2019_challenge(result):
    pos=np.argwhere(result == 1).flatten()
    rpos = []
    pre = 0
    last = len(pos)
    for j in np.where(np.diff(pos) > 40)[0]:
        if j - pre > 40:
            rpos.append(round((pos[pre] + pos[j]) / 2) - 1)
        pre = j + 1

    rpos.append(round((pos[pre] + pos[last-1]) / 2) - 1)

    qrs = np.array(rpos)
    qrs_diff = np.diff(qrs)
    check = True

    while check:
        qrs_diff = np.diff(qrs)
        for r in range(len(qrs_diff)):
            if qrs_diff[r] < 100:
                if result[int(qrs[r])]>result[int(qrs[r+1])]:
                    qrs = np.delete(qrs,r+1)
                    check = True
                    break
                else:
                    qrs = np.delete(qrs,r)
                    check = True
                    break
            check = False

    hr = np.array([loc for loc in qrs if (loc > 2750 and loc < 4750)])

    if len(hr)>1:
        hr = round( 60 * 500 / np.mean(np.diff(hr)))
    else:
        hr = 80

    return hr, qrs
