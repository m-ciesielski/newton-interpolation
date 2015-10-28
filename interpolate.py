import plotly.plotly as py
import argparse
from plotly.graph_objs import Scatter


def parseargs():
    parser = argparse.ArgumentParser(description='Newton\'s Interpolation .')
    parser.add_argument('--start', help='Beginning of X values range.')
    parser.add_argument('--end', help='End of X values range.')
    parser.add_argument('-n', help='Count of nodes.')
    parser.add_argument('--nodes', nargs='+', help='Nodes Y values.')
    return parser.parse_args()


def calculate_nodes(x_start, x_end, nodes_y):
    """
    Calculates X values for given list of Y values in range defined by a and b parameters. X values are simply calculated
    by dividing given X range by number of nodes, so they are distributed in even range.
    :param x_start: Start of X values range
    :param x_end: End of X values range
    :param nodes_y: List of Y values
    :return: List of nodes with X and Y values
    """
    nodes_x = [int(x_start+((x_end-x_start)/(len(nodes_y)-1))*i) for i in range(0, len(nodes_y))]
    nodes_y = [int(y) for y in nodes_y]
    print("X = {0}".format(nodes_x))
    print("Y = {0}".format(nodes_y))
    nodes = zip(nodes_x, nodes_y)
    return list(nodes)


def calculate_newton_interpolation(nodes):
    """
    TODO: implement Newton's interpolation algorithm.
    :param nodes:  List of nodes to interpolate
    """
    pass


def draw_initial_plot(nodes):
    trace = Scatter(x=[node[0] for node in nodes], y=[node[1] for node in nodes])
    data = [trace]
    py.plot(data, filename='plot')


args = parseargs()
args.start = int(args.start)
args.end = int(args.end)
draw_initial_plot(calculate_nodes(args.start, args.end, args.nodes))
