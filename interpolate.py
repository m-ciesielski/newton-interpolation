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
    nodes_x = [float(x_start + ((x_end - x_start) / (len(nodes_y) - 1)) * i) for i in range(0, len(nodes_y))]
    nodes_y = [float(y) for y in nodes_y]
    print(nodes_x)
    print(nodes_y)
    nodes = list(zip(nodes_x, nodes_y))
    return nodes


def calculate_divided_differences_row(nodes_to_compute):
    """
    Takes list of divided differences nodes and calculates new divided differences node from each pair
    of nodes_to_compute.
    In other words, it computes next level of so called Newton's second interpolation form tree.
    :return : list of calculated divided differences
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
    :return : list of tuples of divided differences
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
    :return : String representing calculated polynomial
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


def draw_interpolation_plot(start_x, end_x, interpolation_polynomial, nodes, freq=200, additional_polynomial=None,
                            additional_nodes=None):
    """
    Draws interpolation plot for given interpolation polynomial and nodes.
    """
    # TODO: calculate figure size dynamically
    plt.figure(figsize=(8, 6), dpi=80)
    x = numpy.linspace(start_x, end_x, freq)
    # TODO: eval should be changed to something more secure (like numexpr evaluate())...
    y = eval(str(interpolation_polynomial))
    plt.subplot(211)
    plt.plot(x, y, [node[0] for node in nodes], [node[1] for node in nodes], 'ro')
    plt.grid(True)

    if additional_polynomial:
        poly_values = eval(str(additional_polynomial))
        plt.subplot(212)
        plt.plot(x, poly_values, [node[0] for node in additional_nodes], [node[1] for node in additional_nodes], 'ro')
        plt.grid(True)

    plt.show()


def add_new_node_to_interpolation(polynomial, nodes):
    new_node = nodes[-1]
    # Calculate multiplier
    # TODO: change eval to numexpr evaluate()
    nominator = (float(new_node[1]) - eval(str(polynomial).replace("x", str(new_node[0]))))
    denominator = 1
    for node in nodes[:-1]:
        denominator = denominator * (new_node[0]-node[0])

    multiplier = (nominator/denominator)

    # build new polynomial
    new_interpolation_polynomial = list()
    new_interpolation_polynomial.append("{0}+({1})".format(str(polynomial), multiplier))
    for node in nodes[:-1]:
        new_interpolation_polynomial.append("*(x-{0})".format(node[0]))

    new_interpolation_polynomial_str = "".join(new_interpolation_polynomial)
    print("Calculated polynomial: {0}".format(new_interpolation_polynomial_str))
    new_interpolation_polynomial_str = sy.simplify(new_interpolation_polynomial_str)
    print("Simplified polynomial: {0}".format(new_interpolation_polynomial_str))

    return new_interpolation_polynomial_str


def parse_user_provided_float(label):
    val = None
    while True:
        try:
            val = float(input("Type {0}:".format(label)))
        except ValueError:
            print("Type correct {0} value.".format(label))
            continue
        else:
            break

    return val


def parse_user_provided_float_list(label):
    val_list = list()
    while True:
        try:
            val_list.append(float(input("Type {0}:".format(label))))
        except ValueError:
            response = input("Stop (Y/N)?.".format(label))
            if response == 'Y':
                break
            else:
                continue

    return val_list


def parseargs():
    parser = argparse.ArgumentParser(description='Newton\'s Interpolation .')
    parser.add_argument('--start', type=float, help='Start of X values range.')
    parser.add_argument('--end', type=float, help='End of X values range.')
    parser.add_argument('--nodes-y-values', type=float, nargs='+', help='Y values of interpolation nodes.')
    parsed_args = parser.parse_args()

    if not parsed_args.start:
        parsed_args.start = parse_user_provided_float("start of X values range")

    if not parsed_args.end:
        parsed_args.end = parse_user_provided_float("end of X values range")

    if parsed_args.start >= parsed_args.end:
        print("Range of X values must be greater than 0.")
        exit(2)

    if not parsed_args.nodes_y_values:
        print("Enter Y values of interpolation nodes. Type Y to stop.")
        parsed_args.nodes_y_values = parse_user_provided_float_list("Y value of interpolation node")

    if len(parsed_args.nodes_y_values) < 2:
        print("Provide at least two nodes.")
        exit(3)

    return parsed_args


if __name__ == '__main__':
    args = parseargs()

    args.start = int(args.start)
    args.end = int(args.end)

    init_nodes = prepare_initial_nodes(args.start, args.end, args.nodes_y_values)
    divided_diffs = calculate_divided_differences(init_nodes)
    interpolation_poly = calculate_newton_interpolation(divided_diffs)

    if input("Add new node (Y/N)?") == "Y":
        new_node_x = parse_user_provided_float("X value of new node")
        new_node_y = parse_user_provided_float("Y value of new node")

        new_initial_nodes = list(init_nodes)
        new_initial_nodes.append((float(new_node_x), new_node_y))
        new_polynomial = add_new_node_to_interpolation(interpolation_poly, new_initial_nodes)
        draw_interpolation_plot(start_x=args.start, end_x=args.end, interpolation_polynomial=interpolation_poly,
                                nodes=init_nodes, additional_polynomial=new_polynomial,
                                additional_nodes=new_initial_nodes)
    else:
        draw_interpolation_plot(start_x=args.start, end_x=args.end, interpolation_polynomial=interpolation_poly,
                                nodes=init_nodes)
