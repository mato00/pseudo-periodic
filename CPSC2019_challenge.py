import numpy as np
np.set_printoptions(threshold=np.inf)

def CPSC2019_challenge(result):
    pos=np.argwhere(result == 1).flatten()
    # print (pos)
    rpos = []
    pre = 0
    last = len(pos)-1
    for j in np.where(np.diff(pos) > 20)[0]:
        if j - pre > 20:
            r_cand = round((3*pos[pre] + 2*pos[j]) / 5)
            # print (r_cand)
            rpos.append(r_cand - 1)
        pre = j + 1

    rpos.append(round((3*pos[pre] + 2*pos[last]) / 5))

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
