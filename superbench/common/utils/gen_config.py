from superbench.common.utils import logger

def gen_pair_wise_config(n):
    """Generate pair-wised VM pair config.

    One-to-one means that each participant plays every other participant once.
    The algorithm refers circle method of Round-robin tournament in
    https://en.wikipedia.org/wiki/Round-robin_tournament.
    if n is even, there are a total of n-1 rounds, with n/2 pair of 2 unique participants in each round.
    If n is odd, there will be n rounds, each with n-1/2 pairs, and one participant rotating empty in that round.
    In each round, pair up two by two from the beginning to the middle as (begin, end),(begin+1,end-1)...
    Then, all the participants except the beginning shift left one position, and repeat the previous step.

    Args:
        n (int): the number of participants.

    Returns:
        list: the generated config list, each item in the list is a str like "0,1;2,3".
    """
    config = []
    candidates = list(range(n))
    # Add a fake participant if n is odd
    if n % 2 == 1:
        candidates.append(-1)
    count = len(candidates)
    non_moving = [candidates[0]]
    for _ in range(count - 1):
        pairs = [
            '{},{}'.format(candidates[i], candidates[count - i - 1]) for i in range(0, count // 2)
            if candidates[i] != -1 and candidates[count - i - 1] != -1
        ]
        row = ';'.join(pairs)
        config.append(row)
        robin = candidates[2:] + candidates[1:2]
        candidates = non_moving + robin
    return config


def gen_k_batch_config(scale, n):
    """Generate VM groups config with specified batch scale .

    Args:
        k (int): the scale of batch.
        n (int): the number of participants.

    Returns:
        list: the generated config list, each item in the list is a str like "0,1;2,3".
    """
    config = []
    if  scale <= 0 or n <= 0:
        logger.error('scale and n is not positive')
        return config
    if  scale > n:
        logger.error('scale large than n')
        return config

    group = []
    rem = n % scale
    for i in range(0, n - rem, scale):
        group.append(','.join(map(str, list(range(i, i+scale)))))
    config = [";".join(group)]
    return config