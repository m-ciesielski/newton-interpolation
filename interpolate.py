import sympy as sy
import argparse
from matplotlib import pyplot as plt
from numpy import *


class Node(object):
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
        return Node(left_parent=left_parent, right_parent=right_parent)


def calculate_initial_nodes(x_start, x_end, nodes_y):
    """
    Calculates X values for given list of Y values in range defined by a and b parameters. X values are simply calculated
    by dividing given X range by number of nodes, so they are distributed in even range.
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
    divided_differences = []

    for i in range(0, len(nodes_to_compute) - 1):
        if len(nodes_to_compute) == 1:
            return None
        child = Node.create_child_node(nodes_to_compute[i], nodes_to_compute[i + 1])
        child.calculate_value()
        divided_differences.append(child)

    for node in divided_differences:
        print(node, end='')

    print('\n')
    return divided_differences


def calculate_divided_differences(nodes):
    nodes_to_compute = []
    divided_differences = []
    for node in nodes:
        nodes_to_compute.append(Node(x=node[0], divided_difference=node[1]))

    divided_differences.append(tuple(nodes_to_compute))

    while len(nodes_to_compute) > 1:
        next_node_row = calculate_divided_differences_row(nodes_to_compute)
        divided_differences.append(tuple(next_node_row))
        nodes_to_compute = next_node_row

    return divided_differences


def calculate_newton_interpolation(divided_differences):
    polynomial = []
    for i, divided_differences_row in enumerate(divided_differences):
        polynomial_part = '({0})'.format(divided_differences_row[0].divided_difference)
        for j in range(0, i):
            polynomial_part += '*(x-{0})'.format(divided_differences[0][j].x)

        polynomial_part += '+'

        polynomial.append(polynomial_part)

    polynomial_str = ''.join(polynomial)[:-1]

    print('Calculated polynomial: {0}'.format(polynomial_str))
    simplified_polynomial = sy.simplify(polynomial_str)
    print("Simplified polynomial: {0}".format(simplified_polynomial))
    return simplified_polynomial


def draw_interpolation_plot(interpolation_polynomial=None, initial_nodes=None, start_x=0, end_x=15, freq=200):
    plt.figure(figsize=(8, 6), dpi=80)
    x = linspace(start_x, end_x, freq)
    # eval should be changed to something more secure (like numexpr evaluate())...
    y = eval(str(interpolation_polynomial))
    if initial_nodes:
        initial_x = []
        initial_y = []
        for node in initial_nodes:
            initial_x.append(node[0])
            initial_y.append(node[1])

        plt.plot(x, y, initial_x, initial_y, 'ro')
    else:
        plt.plot(x, y)

    plt.grid(True)
    plt.show()


def calculate_and_draw_newton_interpolation(args=None, nodes=None):
    if nodes:
        initial_nodes = calculate_initial_nodes(args.start, args.end, nodes)
    else:
        initial_nodes = calculate_initial_nodes(args.start, args.end, args.nodes)

    divided_differences = calculate_divided_differences(initial_nodes)
    interpolation_polynomial = calculate_newton_interpolation(divided_differences)
    draw_interpolation_plot(start_x=args.start, end_x=args.end, interpolation_polynomial=interpolation_polynomial,
                            initial_nodes=initial_nodes)


def add_new_node_to_interpolation(polynomial, node):
    """
    TODO: implement addition of node to already calculated polynomial.
    """
    pass


def parseargs():
    parser = argparse.ArgumentParser(description='Newton\'s Interpolation .')
    parser.add_argument('--start', required=True, help='Beginning of X values range.')
    parser.add_argument('--end', required=True, help='End of X values range.')
    parser.add_argument('-n', required=False, help='Count of nodes.')
    parser.add_argument('--nodes', required=True, nargs='+', help='Nodes Y values.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parseargs()
    args.start = int(args.start)
    args.end = int(args.end)

    calculate_and_draw_newton_interpolation(args=args)

    new_node_y = input("Pass new node Y value:")
    new_nodes = args.nodes
    new_nodes.append(new_node_y)

    args.end += 1
    calculate_and_draw_newton_interpolation(args=args, nodes=new_nodes)
