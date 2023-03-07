SRATE = 128  # in hz
SEGLEN = 4 * SRATE  # samples
BATCH_SIZE = 1024
MAX_CASES = 100

cachefile = '{}sec_{}cases.npz'.format(SEGLEN // SRATE, MAX_CASES)
if os.path.exists(cachefile):
    dat = np.load(cachefile)
    x, y, b, c = dat['x'], dat['y'], dat['b'], dat['c']
else:
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")  # track information
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # patient information

    # Column order when loading data
    EEG = 0
    SEVO = 1
    BIS = 2

    # Inclusion & Exclusion criteria
    caseids = set(df_cases.loc[df_cases['age'] > 18, 'caseid']) &\
        set(df_trks.loc[df_trks['tname'] == 'BIS/EEG1_WAV', 'caseid']) &\
        set(df_trks.loc[df_trks['tname'] == 'BIS/BIS', 'caseid']) &\
        set(df_trks.loc[df_trks['tname'] == 'Primus/EXP_SEVO', 'caseid'])

    x = []  
    y = []  # sevo
    b = []  # bis
    c = []  # caseids
    icase = 0  # number of loaded cases
    for caseid in caseids:
        print('loading {} ({}/{})'.format(caseid, icase, MAX_CASES), end='...', flush=True)

        # Excluding the following values
        if np.any(vitaldb.load_case(caseid, 'Orchestra/PPF20_CE') > 0.2):
            print('propofol')
            continue
        if np.any(vitaldb.load_case(caseid, 'Primus/EXP_DES') > 1):
            print('desflurane')
            continue
        if np.any(vitaldb.load_case(caseid, 'Primus/FEN2O') > 2):
            print('n2o')
            continue
        if np.any(vitaldb.load_case(caseid, 'Orchestra/RFTN50_CE') > 0.2):
            print('remifentanil')
            continue

        # Extract data
        vals = vitaldb.load_case(caseid, ['BIS/EEG1_WAV', 'Primus/EXP_SEVO', 'BIS/BIS'], 1 / SRATE)
        if np.nanmax(vals[:, SEVO]) < 1:
            print('all sevo <= 1')
            continue

        # Convert etsevo to the age related mac
        age = df_cases.loc[df_cases['caseid'] == caseid, 'age'].values[0]
        vals[:, SEVO] /= 1.80 * 10 ** (-0.00269 * (age - 40))

        if not np.any(vals[:, BIS] > 0):
            print('all bis <= 0')
            continue

        # Since the EEG should come out well, we start from the location where the value of bis was first calculated.
        valid_bis_idx = np.where(vals[:, BIS] > 0)[0]
        first_bis_idx = valid_bis_idx[0]
        last_bis_idx = valid_bis_idx[-1]
        vals = vals[first_bis_idx:last_bis_idx + 1, :]

        if len(vals) < 1800 * SRATE:  # Do not use cases that are less than 30 minutes
            print('{} len < 30 min'.format(caseid))
            continue

        # Forward fill in MAC value and BIS value up to 5 seconds
        vals[:, SEVO:] = pd.DataFrame(vals[:, SEVO:]).ffill(limit=5 * SRATE).values

        # Extract data every 1 second from its start to its end and then put into the dataset
        oldlen = len(y)
        for irow in range(SEGLEN, len(vals), SRATE):
            bis = vals[irow, BIS]
            mac = vals[irow, SEVO]
            if np.isnan(bis) or np.isnan(mac) or bis == 0:
                continue
            # add dataset
            eeg = vals[irow - SEGLEN:irow, EEG]
            x.append(eeg)
            y.append(mac)
            b.append(bis)
            c.append(caseid)

        # Valid case
        icase += 1
        print('{} samples read -> total {} samples ({}/{})'.format(len(y) - oldlen, len(y), icase, MAX_CASES))
        if icase >= MAX_CASES:
            break

    # Change the input dataset to a numpy array
    x = np.array(x)
    y = np.array(y)
    b = np.array(b)
    c = np.array(c)

    # Save cahce file
    np.savez(cachefile, x=x, y=y, b=b, c=c)
