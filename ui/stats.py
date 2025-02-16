from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QMainWindow, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
import sys

class StatPanel:
    def __init__(self, ui):
        self.ui = ui  # Reference to the main UI

    def dropdowns(self):
        self.handle_selection_1()
        self.handle_selection_2()
        self.handle_selection_3()


    def handle_selection_1(self):
        choice = self.ui.resampling_dropdown.currentIndex()
        # print(choice)

        if choice == 0:
            pass

        elif choice == 1:
            print("Selected Nearest Neighbor")

        elif choice == 2:
            print("Selcted Bilinear Interpolation")

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
        choice = self.ui.options_dropdown.currentIndex()
        # print(choice)

        if choice == 0:
            pass

        elif choice == 1:
            self.ui.graph_1.setVisible(True)
            self.ui.graph_2.setVisible(True)
            # self.display_image(self.ui.graph_1, "/home/brian/research/results/brf_analysis/BRF_Comparison_Band_1.png")
            # self.display_image(self.ui.graph_2, "/home/brian/research/results/brf_analysis/BRF_Comparison_Band_2.png")
            print("Selected Side-by-Side")

        elif choice == 2:
            print("Selected Difference")

    def display_image(self, view, image_path):
        """Loads and displays an image in the specified QGraphicsView."""
        scene = QGraphicsScene()  # Create a new scene
        pixmap = QPixmap(image_path)  # Load the image
        pixmap_item = QGraphicsPixmapItem(pixmap)  # Convert to pixmap item
        scene.addItem(pixmap_item)  # Add to scene
        view.setScene(scene)  # Set scene to the view
        



