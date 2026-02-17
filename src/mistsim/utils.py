def get_lmax(alm_shape):
    """
    Get the maximum l value from the alm array in the S2FFT format.

    Parameters
    ----------
    alm : tup
        The shape of an alm array in the S2FFT format. The last two
        axes should correspond to the l and m values, respectively. Any
        previous axes are considered as batch dimensions.

    Returns
    -------
    lmax : int
        The maximum l value.

    """
    # each row corresponds to a different ell, starting from 0
    lmax = alm_shape[-2] - 1
    return lmax
