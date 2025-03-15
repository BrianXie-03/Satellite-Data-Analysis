from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QMainWindow, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
import scipy
import sys

class StatPanel:
    def __init__(self, ui):
        self.ui = ui  # Reference to the main UI

    def dropdowns(self):
        self.handle_selection_1()
        self.handle_selection_2()
        self.handle_selection_3()


    def handle_selection_1(self):
        # choice = self.ui.resampling_dropdown.currentIndex()
        # zoom_factors = (new_shape[0] / data.shape[0], new_shape[1] / data.shape[1])
        # # print(choice)

        # if choice == 0:
        #     pass

        # elif choice == 1:
        #     return scipy.ndimage.zoom(data, zoom_factors, order=0)

        # elif choice == 2:
        #     return scipy.ndimage.zoom(data, zoom_factors, order=1)
        # else:
        #     raise ValueError("Method must be 'nearest' or 'bilinear'")
        return 1

    def handle_selection_2(self):
        choice = self.ui.metric_dropdown.currentIndex()
        # print(choice)

        if choice == 0:
            pass

        elif choice == 1:
            print("Selected RMSE")

        elif choice == 2:
            print("Selcted Correlation")
        
        elif choice == 3:
            print("Selected Standard Deviation")
    
    def handle_selection_3(self):
        self.ui.graph_1.setVisible(False)
        self.ui.graph_2.setVisible(False)
        self.ui.difference_graph.setVisible(False)
        choice = self.ui.options_dropdown.currentIndex()
        
        if choice == 0:
            pass

        elif choice == 1:
            self.ui.graph_1.setVisible(True)
            self.ui.graph_2.setVisible(True)
            self.display_image(self.ui.graph_1, "/home/brian/research/results/brf_analysis/BRF_Comparison_ref.png")
            self.display_image(self.ui.graph_2, "/home/brian/research/results/brf_analysis/BRF_Comparison_new.png")
            
            print("Selected Side-by-Side")

        elif choice == 2:
            self.ui.difference_graph.setVisible(True)
            self.display_image(self.ui.difference_graph, "/home/brian/research/results/brf_analysis/BRF_Comparison_diff.png")
            print("Selected Difference")

    def display_image(self, view, image_path):
        """Loads and displays an image in the specified QGraphicsView."""
        scene = QGraphicsScene()
        pixmap = QPixmap(image_path)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(pixmap_item)
        view.setScene(scene)
        



