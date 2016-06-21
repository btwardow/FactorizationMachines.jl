import sys


with open(sys.argv[1], 'r') as f_in:
    with open(sys.argv[2], 'w') as f_train:
        with open(sys.argv[3], 'w') as f_test:
            for i, l in enumerate(f_in):
                f_out = f_train if i % 5 else f_test
                u, m, r, _ = l.split('::')
                f_out.write('{} {}:1 {}:1\n'.format(r, u, 71567 + int(m)))
