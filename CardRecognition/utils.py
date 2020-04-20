
def card_gen():
    l1 = '23456789tjqka'
    l2 = 'dchs'

    c = []
    for i in l1:
        for j in l2:
            tc = i + j
            c.append(tc)

    print(c)
