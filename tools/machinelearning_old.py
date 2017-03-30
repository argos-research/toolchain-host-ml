import time

from matplotlib import cm
from matplotlib.cbook import Null
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pip._vendor.pyparsing import line
from sklearn import linear_model
from sklearn import svm
import sklearn
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import numpy as np


# from mpl_toolkits.mplot3d import Axes3D
if False: plt = None


class machinelearning:
    input_2dvector = None  # Input parameter
    output_1dvector = None  # Output parameter
    data_per_set = 0  # amount of values per training tuple
    X_new = None

    # Vector Machines 
    svc = Null 
    rbf_svc = Null
    poly_svc = Null
    lin_svc = Null
    sgd = Null
    # title for the plots
    titles = []
    
   
    def __init__(self, data_per_set, AmountRows):
        self.input_2dvector = np.ndarray(shape=(AmountRows, data_per_set - 1))  # Input parameter
        self.output_1dvector = np.ndarray(shape=[AmountRows])  # Output parameter
        self.data_per_set = data_per_set - 1  # dimensions of plot and data

        self.X_new = Null
        # Vector Machines 
        self.svc = Null 
        self.rbf_svc = Null
        self.poly_svc = Null
        self.lin_svc = Null
        self.sgd = Null
        # title for the plots
        self.titles = [
                  'LinearSVC with linear kernel)',  # title for the plots
                  'SVC with poly kernel',
                  'SVC with RBF kernel',
                  'SVC with poly and RBF',
                  'SDG']

    # output method
    def output(self, msg):
        for line in msg.splitlines():
            print "ML: " + line
        

    # stores data in appropriate arrays input_2dvector and output_1dvector
    # @param data: data array with in- and output
    # @param dimensions: 2 for 2d, 3 for 3d
    def training_phase(self, data_input, data_output, CC=1, gamma=0.7, degree=3):
        self.output("Load training data")
        # get data
        self.input_2dvector = np.array(data_input)
        self.output_1dvector = np.ravel(np.array(data_output))
        
        self.output("Train machine learning")
        
        
        # data fitting with time measurement 
        start = time.time()
        indv_start = start
        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        # self.svc = svm.SVC(kernel='linear', cache_size=7000, C=CC).fit(self.input_2dvector, self.output_1dvector)
        indv_end = time.time()
        self.output("1st done: " + str(indv_end - indv_start) + " s")
        indv_start = time.time()
        self.lin_svc = svm.LinearSVC(dual=False, tol=0.00001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight={0: 9}, verbose=0, random_state=None, max_iter=1000000).fit(self.input_2dvector, self.output_1dvector)
        indv_end = time.time()
        self.output("2nd done: " + str(indv_end - indv_start) + " s")
        indv_start = time.time()
        self.rbf_svc = svm.SVC(kernel='rbf', cache_size=7000, tol=0.00001, class_weight={0: 20}, gamma=0.01, C=1).fit(self.input_2dvector, self.output_1dvector)
        indv_end = time.time()
        self.output("3rd done: " + str(indv_end - indv_start) + " s")
        indv_start = time.time()
        self.poly_svc = svm.SVC(kernel='poly', degree=2, cache_size=7000, tol=0.0001, class_weight={1: 20}, C=CC).fit(self.input_2dvector, self.output_1dvector)
        indv_end = time.time()
        self.output("4th done: " + str(indv_end - indv_start) + " s")
        end = time.time()
        
        self.sgd = linear_model.SGDClassifier().fit(self.input_2dvector, self.output_1dvector, np.unique(self.output_1dvector))
        
        self.output("Training time for " + str(len(self.output_1dvector)) + " sets: " + str(end - start))


    def save_training(self, filename):
        self.output("Save training set")
        joblib.dump(self.lin_svc, filename + "_lin_svc.pkl")
        joblib.dump(self.rbf_svc, filename + "_rbf.pkl")
        joblib.dump(self.poly_svc, filename + "_poly.pkl")
        joblib.dump(self.sgd, filename + "_sgd.pkl")
        
    def load_training(self, filename):
        self.output("Load training set")
        self.lin_svc = joblib.load(filename + "_lin_svc.pkl") 
        self.rbf_svc = joblib.load(filename + "_rbf.pkl") 
        self.poly_svc = joblib.load(filename + "_poly.pkl")
        self.sgd = joblib.load(filename + "_sgd.pkl") 
        
        

    # plots 2D plot, throws error for 3D data
    def plot2D(self, h):
        # create a mesh to plot in, only if |input_2dvector data|==2
        x_min, x_max = self.input_2dvector[:, 0].min() - 50, self.input_2dvector[:, 0].max() + 50
        y_min, y_max = self.input_2dvector[:, 1].min() - 50, self.input_2dvector[:, 1].max() + 50
        
        # only plot results, if data equals 2. In other cases no proper visualisation is possible
        if(self.data_per_set == 2):
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        else:
            self.output("data_per_set not equal 2. Abort plotting")
            return
            
        # get predicted data for three kernels
        Z_lin = self.lin_svc.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_poly = self.poly_svc.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_rbf = self.rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_SGD = self.sgd.predict(np.c_[xx.ravel(), yy.ravel()])
        
        self.output("Start ploting")
        
        for i, Z_ in enumerate((Z_lin, Z_poly, Z_rbf, Z_rbf, Z_SGD)):
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            plt.subplot(2, 3, i + 1)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
        
            # Visualization of ml-function, only if |input_2dvector data|==2
            if(self.data_per_set == 2):
                if(i == 3):  # catch linear svc and replace it with a combination of rbf and lin/poly                    
                    for j in range(len(Z_)):
                        if(Z_[j] + Z_poly[j] != 2):
                            Z_[j] = 0
                                            
                # Put the result into a color plot
                Z = Z_.reshape(xx.shape)
                plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm_r, alpha=0.8)
                plt.xlim(xx.min(), xx.max())
                plt.ylim(yy.min(), yy.max())
            else:
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)

            # Plot also the training points
            plt.scatter(self.input_2dvector[:, 0], self.input_2dvector[:, 1], c=self.output_1dvector, cmap=plt.cm.coolwarm_r, alpha=0.5)
            plt.xlabel('Period Task 1')
            plt.ylabel('Period Task 2')
            plt.xticks(())
            plt.yticks(())
            plt.title(self.titles[i])
            
        self.output("Plotting  done")
        plt.show()
    

        
# plots 3D plot, throws error for 2D data    
    def plot3D(self, h):        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # map points to red and blue
        def outputc(x):
            if(x == 0.0):
                return 'r'
            else:
                return 'b'
        color = map(outputc, self.output_1dvector)
        # print points
        ax.scatter(self.input_2dvector[:, 0], self.input_2dvector[:, 1], self.input_2dvector[:, 2], c=color, marker='o')
        
        # no surface plotting for 3D        
        # create a mesh to plot in        
        '''x_min, x_max = self.input_2dvector[:, 0].min() - 1, self.input_2dvector[:, 0].max() + 1
        y_min, y_max = self.input_2dvector[:, 1].min() - 1, self.input_2dvector[:, 1].max() + 1
        z_min, z_max = self.input_2dvector[:, 2].min() - 1, self.input_2dvector[:, 2].max() + 1
        

        xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h),
                                 np.arange(z_min, z_max, h))

        Z = self.svc.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
        Z = Z.reshape(xx.shape)
        
        
                
        # plot random surface function, maybe useful in future
        input_2dvector = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        input_2dvector, Y = np.meshgrid(input_2dvector, Y)
        R = np.sqrt(input_2dvector**2 + Y**2)
        Z = np.sin(R)

        
        # Plot the surface.
        surf = ax.plot_surface(input_2dvector, Y, Z, cmap=cm.coolwarm, alpha=0.2,
                               linewidth=0, antialiased=False)
    
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
                
        # Customize the z axis.
        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))'''
        


        ax.set_xlabel('input_2dvector Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.title("SVC with linear kernel")

        plt.show()

    def plot(self, h=0.2, dimensions=2):
        self.output("Plotting data with h = " + str(h))
        if(dimensions == 2):
            self.plot2D(h)
        elif(dimensions == 3):
            self.plot3D(h)
        else:
            raise Exception("Wrong dimension for plotting!")
            
