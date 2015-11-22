import sympy as sy
import argparse
from matplotlib import pyplot as plt
import numpy


class DividedDifferenceNode(object):
    """
    Represents node in divided differences tree.
    """
    divided_difference = None
    x = None
    left_parent = None
    right_parent = None

    def __init__(self, left_parent=None, right_parent=None, x=None, divided_difference=None):
        self.left_parent = left_parent
        self.right_parent = right_parent
        self.x = x
        self.divided_difference = divided_difference

    def __str__(self):
        return "Value: {0},".format(self.divided_difference)

    def calculate_value(self):
        self.divided_difference = ((self.right_parent.divided_difference - self.left_parent.divided_difference) / (
            self.get_right_x() - self.get_left_x()))

    def get_left_x(self):
        if self.x is not None:
            return self.x
        else:
            return self.left_parent.get_left_x()

    def get_right_x(self):
        if self.x:
            return self.x
        else:
            return self.right_parent.get_right_x()

    @staticmethod
    def create_child_node(left_parent, right_parent):
        return DividedDifferenceNode(left_parent=left_parent, right_parent=right_parent)


def prepare_initial_nodes(x_start, x_end, nodes_y):
    """
    Calculates X values for given list of Y values in range defined by a and b parameters. X values are
     simply calculated by dividing given X range by number of nodes, so they are distributed in even range.
    :param x_start: Start of X values range
    :param x_end: End of X values range
    :param nodes_y: List of Y values
    :return: List of nodes with X and Y values
    """
    nodes_x = [int(x_start + ((x_end - x_start) / (len(nodes_y) - 1)) * i) for i in range(0, len(nodes_y))]
    nodes_y = [int(y) for y in nodes_y]
    print(nodes_x)
    print(nodes_y)
    nodes = list(zip(nodes_x, nodes_y))
    return nodes


def calculate_divided_differences_row(nodes_to_compute):
    """
    Takes list of divided differences nodes and calculates new divided differences node from each pair
    of nodes_to_compute.
    In other words, it computes next level of so called Newton's second interpolation form tree.
    :rtype : list of calculated divided differences
    """
    divided_differences = []

    if len(nodes_to_compute) == 1:
        return None

    for i in range(0, len(nodes_to_compute) - 1):
        child = DividedDifferenceNode.create_child_node(nodes_to_compute[i], nodes_to_compute[i + 1])
        child.calculate_value()
        divided_differences.append(child)

    for node in divided_differences:
        print(node, end='')

    print('\n')
    return divided_differences


def calculate_divided_differences(nodes):
    """
    Calculates divided differences for given interpolation nodes.
    It is assumed, that at least two interpolation nodes are provided.
    Each tuple of returned list represents one level of divided differences tree.
    :rtype : list of tuples of divided differences
    """
    nodes_to_compute = []
    divided_differences = []
    for node in nodes:
        nodes_to_compute.append(DividedDifferenceNode(x=node[0], divided_difference=node[1]))

    divided_differences.append(tuple(nodes_to_compute))

    while len(nodes_to_compute) > 1:
        next_node_row = calculate_divided_differences_row(nodes_to_compute)
        divided_differences.append(tuple(next_node_row))
        nodes_to_compute = next_node_row

    return divided_differences


def calculate_newton_interpolation(divided_differences):
    """
    Creates polynomial from given list of divided differences. Polynomial string is created according to equation
    provided in project docs.
    :rtype : String representing calculated polynomial
    """
    polynomial = []
    for i, divided_differences_row in enumerate(divided_differences):
        polynomial_part = '({0})'.format(divided_differences_row[0].divided_difference)
        for j in range(0, i):
            polynomial_part += '*(x-{0})'.format(divided_differences[0][j].x)

        polynomial_part += '+'
        polynomial.append(polynomial_part)
    polynomial_str = ''.join(polynomial)[:-1]

    print('Calculated polynomial: {0}'.format(polynomial_str))
    # Heuristic simplification of calculated polynomial
    simplified_polynomial = sy.simplify(polynomial_str)
    print("Simplified polynomial: {0}".format(simplified_polynomial))
    return simplified_polynomial


def draw_interpolation_plot(interpolation_polynomial=None, initial_nodes=None, start_x=0, end_x=15, freq=200):
    """
    Draws interpolation plot for given interpolation polynomial and nodes.
    """

    # TODO: calculate figure size dynamically
    plt.figure(figsize=(8, 6), dpi=80)
    x = numpy.linspace(start_x, end_x, freq)
    # TODO: eval should be changed to something more secure (like numexpr evaluate())...
    y = eval(str(interpolation_polynomial))
    initial_x = []
    initial_y = []
    for node in initial_nodes:
        initial_x.append(node[0])
        initial_y.append(node[1])

    plt.plot(x, y, initial_x, initial_y, 'ro')

    plt.grid(True)
    plt.show()


def add_new_node_to_interpolation(polynomial, nodes):
    new_node = nodes[-1]
    # Calculate multiplier
    # TODO: change eval to numexpr evaluate()
    nominator = (new_node[1] - eval(str(polynomial).replace("x", str(new_node[0]))))
    denominator = 1
    for node in nodes[:-1]:
        denominator = denominator * (new_node[0]-node[0])

    multiplier = (nominator/denominator)

    # build new polynomial
    new_interpolation_polynomial = list()
    new_interpolation_polynomial.append("{0}+{1}".format(str(polynomial), multiplier))
    for node in nodes[:-1]:
        new_interpolation_polynomial.append("*(x-{0})".format(node[0]))

    new_interpolation_polynomial_str = "".join(new_interpolation_polynomial)
    print("Calculated polynomial: {0}".format(new_interpolation_polynomial_str))
    new_interpolation_polynomial_str = sy.simplify(new_interpolation_polynomial_str)
    print("Simplified polynomial: {0}".format(new_interpolation_polynomial_str))

    return new_interpolation_polynomial_str


def parseargs():
    parser = argparse.ArgumentParser(description='Newton\'s Interpolation .')
    parser.add_argument('--start', required=True, help='Beginning of X values range.')
    parser.add_argument('--end', required=True, help='End of X values range.')
    parser.add_argument('-n', required=False, help='Count of nodes.')
    parser.add_argument('--nodes', required=True, nargs='+', help='Nodes Y values.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parseargs()

    if args.start >= args.end:
        print("Range of X values must be greater than 0.")
    if len(args.nodes) < 1:
        print("Provide at least two nodes.")
    args.start = int(args.start)
    args.end = int(args.end)

    init_nodes = prepare_initial_nodes(args.start, args.end, args.nodes)
    divided_diffs = calculate_divided_differences(init_nodes)
    interpolation_poly = calculate_newton_interpolation(divided_diffs)
    draw_interpolation_plot(start_x=args.start, end_x=args.end, interpolation_polynomial=interpolation_poly,
                            initial_nodes=init_nodes)

    new_node_y = input("Pass new node Y value:")
    new_x_range = args.end + 1
    new_initial_nodes = init_nodes
    new_initial_nodes.append((new_x_range, int(new_node_y)))
    new_polynomial = add_new_node_to_interpolation(interpolation_poly, new_initial_nodes)
    draw_interpolation_plot(start_x=args.start, end_x=new_x_range, interpolation_polynomial=new_polynomial,
                            initial_nodes=new_initial_nodes)
