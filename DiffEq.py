import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
import copy
import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar


# class for the differential equation
class Equation:
    def __init__(self, x0, y0, X, n, solution, f):
        self.x0 = x0
        self.y0 = y0
        self.X = X
        self.n = n
        # the analytical solution of the equation
        self.solution = solution
        # the function f in y' = f(x,y)
        self.f = f


# the parent class for methods of solution
class Method:
    # the function for deriving h, x in all steps and y0
    def solve(self, eq):
        h = (eq.X - eq.x0)/(eq.n - 1)
        x = np.linspace(eq.x0, eq.X, eq.n)
        y = np.zeros(x.shape[0])
        y[0] = eq.y0
        return h, x, y

    # the function for computing the LTE for the method
    def LTE(self, x, eq):
        LTE = np.zeros(x.shape[0])
        # compute the exact solution
        _, y_exact, _ = Exact().solve(eq)
        for i in range(1, x.shape[0]):
            # compute new y_appr by this method using x and y_exact from the previous step
            eq_appr = Equation(x[i-1], y_exact[i-1], x[i], 2, eq.solution, eq.f)
            _, appr, _ = self.solve(eq_appr)
            # compute the LTE
            LTE[i] = abs(y_exact[i] - appr[-1])
        return LTE

    # compute the maximum of the GTE for every n from no to N
    def GTE(self, eq, n0, N):
        # copy the given equation
        eq_appr = copy.deepcopy(eq)
        ans = np.zeros(N-n0+1)
        for i in range(n0, N+1):
            # change the number of steps of the equation
            eq_appr.n = i
            # get the solution by the method
            _, y, _ = self.solve(eq_appr)
            # get the exact solution
            _, y_exact, _ = Exact().solve(eq_appr)
            # compute the maximum GTE
            ans[i-n0] = max(abs(y - y_exact))
        return ans


# the class for exact solution
class Exact(Method):
    def solve(self, eq):
        # retrieving x, y from the parent class' function
        # we don't need h because it's analytical solution
        _, x, y = super().solve(eq)
        # deriving a constant which satisfies the given initial condition
        C = (eq.y0 - np.exp(eq.x0))/np.exp(-eq.x0)
        # return x, solution, method
        return x, eq.solution(x, C), "Exact Solution"


# the class for Euler's method
class Euler(Method):
    def solve(self, eq):
        # retrieving h, x, y from the parent class' function
        h, x, y = super().solve(eq)
        # find y_appr
        for i in range(1, x.shape[0]):
            y[i] = y[i - 1] + h * eq.f(x[i - 1], y[i - 1])
        # return x, solution, method
        return x, y, "Euler's method"


# the class for Improved Euler's method
class ImprovedEuler(Method):
    def solve(self, eq):
        # retrieving h, x, y from the parent class' function
        h, x, y = super().solve(eq)
        # find y_appr
        for i in range(1, x.shape[0]):
            k1 = eq.f(x[i - 1], y[i - 1])
            k2 = eq.f(x[i], y[i - 1] + h * k1)
            y[i] = y[i - 1] + h / 2. * (k1 + k2)
        # return x, solution, method
        return x, y, "Improved Euler's method"


# the class for Runge-Kutta method
class RungeKutta(Method):
    def solve(self, eq):
        # retrieving h, x, y from the parent class' function
        h, x, y = super().solve(eq)
        # find y_appr
        for i in range(1, x.shape[0]):
            k1 = eq.f(x[i - 1], y[i - 1])
            k2 = eq.f(x[i - 1] + h / 2., y[i - 1] + h / 2. * k1)
            k3 = eq.f(x[i - 1] + h / 2., y[i - 1] + h / 2. * k2)
            k4 = eq.f(x[i], y[i - 1] + h * k3)
            y[i] = y[i - 1] + h / 6. * (k1 + 2 * k2 + 2 * k3 + k4)
        # return x, solution, method
        return x, y, "Runge-Kutta method"


# the class for the main window of our application
class MyWindow(QMainWindow):
    # for the initialization we need the equation, all methods and parameters no and N
    def __init__(self, eq, methods, n0, N, ):
        super(MyWindow, self).__init__()
        self.controller = Controller(eq, n0, N, methods, [True, True, True, True], self)
        # initialization of fields for user input of x0, y0, n, X, n0 , N and filling it with initial values
        # QLineEdit means a field for input the values
        self.lineEdit_x0 = QLineEdit('0')
        self.lineEdit_y0 = QLineEdit('0')
        self.lineEdit_n = QLineEdit('8')
        self.lineEdit_X = QLineEdit('7')
        self.lineEdit_n0 = QLineEdit('8')
        self.lineEdit_N = QLineEdit('71')
        # initialization of the list of lineEdits
        self.lineEdits = [self.lineEdit_x0, self.lineEdit_y0, self.lineEdit_n, self.lineEdit_X, self.lineEdit_n0,
                          self.lineEdit_N]
        # initialization of the list of the labels
        labels = [QLabel('x0'),  QLabel('y0'), QLabel('n'), QLabel('X'), QLabel('n0'), QLabel('N')]
        # set this validator that allows user to type only integer for all lineEdits
        for lineEdit in self.lineEdits:
            lineEdit.setValidator(QtGui.QIntValidator())
        # initialization of checkboxes for each method
        # if the checkbox for the method is checked, then we need to plot this method on the graph and vice versa
        # initialization of the list of checkboxes
        self.checkBoxes = [QCheckBox("Exact Solution", self), QCheckBox("Euler's method", self),
                           QCheckBox("Improved Euler's method", self), QCheckBox("Runge-Kutta method", self)]
        # for every checkbox we need to set its initial state to "Checked"
        for checkBox in self.checkBoxes:
            checkBox.setChecked(True)
        # button for replotting graphs
        button = QPushButton('Plot')
        # create the object for 3 tabs
        self.table_widget = MyTableWidget(self)
        # we need QHBoxLayouts and QVBoxLayouts to construct horizontal and vertical box layout objects correspondingly
        # the general layout
        layout = QHBoxLayout()
        # layout for changing the parameters and replotting the graphs
        layout_parameters = QVBoxLayout()
        # layout for user input (x0, y0, n, X, n0, N)
        layout_input = QHBoxLayout()
        # layout for labels
        layout_labels = QVBoxLayout()
        # layout for lineEdits
        layout_lineEdits = QVBoxLayout()
        # layout for checkBoxes
        layout_checkBoxes = QVBoxLayout()
        # we need to add to the general layout our tabs
        layout.addWidget(self.table_widget)
        # we need to add every label to the layout_labels and lineEdit to the layout_lineEdits
        for label, lineEdit in zip(labels, self.lineEdits):
            layout_labels.addWidget(label)
            layout_lineEdits.addWidget(lineEdit)
        # we need to add every checkBox to the layout_checkBoxes
        for checkBox in self.checkBoxes:
            layout_checkBoxes.addWidget(checkBox)
        # layout_labels is the part of the layout_input so we need to add it to the layout_input
        layout_input.addLayout(layout_labels)
        # analogically, we need to add the layout_Edits to the layout_input
        layout_input.addLayout(layout_lineEdits)
        # layout_input is the part of the layout_parameters so we need to add it to layout_parameters
        layout_parameters.addLayout(layout_input)
        # analogically, we need to find layout_checkBoxes to the layout_parameters
        layout_parameters.addLayout(layout_checkBoxes)
        # we need to add button for replotting to the layout_parameters
        layout_parameters.addWidget(button)
        # widget for fixing the width of the layout_parameters
        widget_parameters = QWidget()
        # we need to add the layout_parameters into the widget_parameters
        widget_parameters.setLayout(layout_parameters)
        # set width of the widget_parameters
        widget_parameters.setFixedWidth(200)
        # we need to add the widget_parameters to the general layout
        layout.addWidget(widget_parameters)
        # the general widget
        widget = QWidget()
        # set the general layout as the layout for the general widget
        widget.setLayout(layout)
        # if the button is clicked, we need to change parameters of the equation and to replot the graph
        button.clicked.connect(self.controller.change_parameters)
        # if some checkBox changes its state, we need to add/remove the corresponding method to the graph
        for checkBox in self.checkBoxes:
            checkBox.stateChanged.connect(self.controller.moderateMethods)
        # set the general widget as the central widget
        self.setCentralWidget(widget)
        # plot the initial graph
        self.controller.update()


# the class fot the tabs
class MyTableWidget(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        # colours for the methods in the graph
        self.colours = ['b-', 'g-', 'r-', 'k-']
        # initialize the figures for the graphs
        figure1 = plt.figure()
        figure2 = plt.figure()
        figure3 = plt.figure()
        self.figures = [figure1, figure2, figure3]
        # the layout for graphs
        layout = QVBoxLayout(self)
        # the tab widget for the tabs
        tabs = QTabWidget()
        # initialization of tabs
        tab_solutions = QWidget()
        tab_LTE = QWidget()
        tab_GTE = QWidget()
        # resize the tab widget
        tabs.resize(300, 200)
        # add tabs to the tab widget
        tabs.addTab(tab_solutions, "Solutions")
        tabs.addTab(tab_LTE, "LTE")
        tabs.addTab(tab_GTE, "GTE")
        # create the Box Layout for the tab with the solutions
        tab_solutions.layout = QVBoxLayout(self)
        # create the canvas the figure of the solutions renders into
        self.canvas_solutions = FigureCanvasQTAgg(figure1)
        # add the canvas and the toolbar to the layout of the tab with the solutions
        tab_solutions.layout.addWidget(NavigationToolbar(self.canvas_solutions, self))
        tab_solutions.layout.addWidget(self.canvas_solutions)
        # set the layout for the tab with the solutions
        tab_solutions.setLayout(tab_solutions.layout)
        # create Box Layout for the tab with the LTE
        tab_LTE.layout = QVBoxLayout(self)
        # create the canvas the figure with the LTE renders into
        self.canvas_LTE = FigureCanvasQTAgg(figure2)
        # add the canvas and the toolbar to the layout of the tab with the LTE
        tab_LTE.layout.addWidget(NavigationToolbar(self.canvas_LTE, self))
        tab_LTE.layout.addWidget(self.canvas_LTE)
        # set the layout for the tab with the LTE
        tab_LTE.setLayout(tab_LTE.layout)
        # create Box Layout for the tab with the change of the GTE maximum
        tab_GTE.layout = QVBoxLayout(self)
        # create the canvas the figure with the change of the GTE maximum renders into
        self.canvas_GTE = FigureCanvasQTAgg(figure3)
        # add the canvas and the toolbar to the layout of the tab with the change of the GTE maximum
        tab_GTE.layout.addWidget(NavigationToolbar(self.canvas_GTE, self))
        tab_GTE.layout.addWidget(self.canvas_GTE)
        # set the layout for the tab with the the change of the GTE maximum
        tab_GTE.setLayout(tab_GTE.layout)
        # add the tabs to the general layout
        layout.addWidget(tabs)
        # set layout for the class
        self.setLayout(layout)

    # function for replotting the graphs
    def plot(self, results, n0, N):
        axes = []
        # clear all the figures and add new subplots
        for figure in self.figures:
            figure.clear()
            axes.append(figure.add_subplot(111))
        for (plotMethod, x, y, label, lte, gte), colour in zip(results, self.colours):
            # if we don't need to plot this method, just skip it
            if plotMethod is False:
                continue
            # plot the results of the method
            axes[0].plot(x, y, colour, label=label)
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            axes[0].legend()
            # plot LTe of the second tab
            axes[1].plot(x, lte, colour, label=label)
            axes[1].legend()
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("LTE")
            # plot the maximum of GTE on the third tab
            axes[2].plot(np.arange(n0, N + 1), gte, colour, label=label)
            axes[2].legend()
            axes[2].set_xlabel("n")
            axes[2].set_ylabel("The maximum of GTE")
        # show the graphs
        self.canvas_solutions.draw()
        self.canvas_LTE.draw()
        self.canvas_GTE.draw()


# the Controller
class Controller:
    # initialization of the class
    def __init__(self, eq, n0, N, methods, plotMethods, window):
        self.eq = eq
        self.n0 = n0
        self.N = N
        self.methods = methods
        self.plotMethods = plotMethods
        self.window = window

    # the function for changing the parameters of the equation and replotting the graph
    def change_parameters(self):
        # if user did not fill some parameter, we should print the error
        for lineEdit in self.window.lineEdits:
            if lineEdit.text() == '':
                QMessageBox.about(self.window, "Error", "You did not fill all items!")
                return
        # read all the user input
        x0 = int(self.window.lineEdit_x0.text())
        y0 = int(self.window.lineEdit_y0.text())
        n = int(self.window.lineEdit_n.text())
        X = int(self.window.lineEdit_X.text())
        n0 = int(self.window.lineEdit_n0.text())
        N = int(self.window.lineEdit_N.text())
        # if number of steps is less than 2, we should print the error
        if n < 2:
            QMessageBox.about(self.window, "Error", "n should be not less than 2!")
            return
        # the same if the x0 is not less than X
        if x0 >= X:
            QMessageBox.about(self.window, "Error", "X should be greater than x0!")
            return
        # if n0 is less than 2, we need to print the error
        if n0 < 2:
            QMessageBox.about(self.window, "Error", "n0 should be not less than 2!")
            return
        # if N is not greater than n0, we need to print the error
        if N <= n0:
            QMessageBox.about(self.window, "Error", "N should be greater than n0!")
            return
        # change all parameters of our equation, n0 and N
        self.eq.x0 = x0
        self.eq.y0 = y0
        self.eq.n = n
        self.eq.X = X
        self.n0 = n0
        self.N = N
        # update the results
        self.update()

    # the method for adding the methods to the graphs / removing it
    def moderateMethods(self, state):
        # if the user wants to add some method to the graphs
        if state == QtCore.Qt.Checked:
            # find the method the user wants to add
            for i in range(len(self.window.checkBoxes)):
                if self.window.sender() == self.window.checkBoxes[i]:
                    # set that we need to plot this method
                    self.plotMethods[i] = True
        # if the user wants to remove some method from the graph
        else:
            # find the method the user wants to remove
            for i in range(len(self.window.checkBoxes)):
                if self.window.sender() == self.window.checkBoxes[i]:
                    # set that we don't need to plot this graph
                    self.plotMethods[i] = False
        # update the results
        self.update()

    # the function for updating the results
    def update(self):
        results = []
        for method, plotMethod in zip(self.methods, self.plotMethods):
            # apply this method
            x, y, label = method.solve(self.eq)
            # compute LTE and the maximum of GTE
            lte = method.LTE(x, self.eq)
            gte = method.GTE(self.eq, self.n0, self.N)
            results.append((plotMethod, x, y, label, lte, gte))
        self.window.table_widget.plot(results, self.n0, self. N)


# initialize our equation
x0 = 0
y0 = 0
solution = lambda x, C: np.exp(x) + C * np.exp(-x)
f = lambda x, y: 2*np.exp(x)-y
X = 7
n = 8
eq = Equation(x0, y0, X, n, solution, f)
app = QApplication([])
app.setApplicationName("DE Calculator")
methods = [Exact(), Euler(), ImprovedEuler(), RungeKutta()]
application = MyWindow(eq, methods, 8, 71)
# run the application
application.show()
sys.exit(app.exec())
