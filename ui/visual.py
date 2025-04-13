from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QMainWindow, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
import scipy
import sys

class VisualsPanel:
    def __init__(self, ui):
        self.ui = ui  # Reference to the main UI
    
    def visual_selection(self):
        self.handle_selection_1()
        self.handle_selection_2()

    def handle_selection_1(self):
        self.ui.vis_graph_diff.setVisible(False)
        self.ui.vis_graph_new.setVisible(False)
        self.ui.vis_graph_ref.setVisible(False)
        choice = self.ui.visualType_dropdown.currentIndex()
        type = self.ui.vis_compare_opt.currentIndex() # 1 is side by side, 2 is diff

        if choice == 0:
            pass

        elif choice == 1:
            self.ui.vis_graph_diff.setVisible(True)
            self.display_image(self.ui.vis_graph_diff, "/home/brian/research/results/brf_analysis/scatterplot_comparison.png")
            print("Selected Scatter Plot")

        elif choice == 2:
            print("Selcted Line Graph")
        
        elif choice == 3:

            if type == 1:
                self.ui.vis_graph_new.setVisible(True)
                self.ui.vis_graph_ref.setVisible(True)
                self.display_image(self.ui.vis_graph_new, "/home/brian/research/results/brf_analysis/QC_Histogram_New.png")
                self.display_image(self.ui.vis_graph_ref, "/home/brian/research/results/brf_analysis/QC_Histogram_Reference.png")

            elif type == 2:
                self.ui.vis_graph_diff.setVisible(True)
                self.display_image(self.ui.vis_graph_diff, "/home/brian/research/results/brf_analysis/QC_Histogram_Difference.png")

            else:
                pass

            print("Selected Histogram")
    
    def handle_selection_2(self):
        print(2)


    def color_schemes(self):
        # choice = self.ui.scheme_dropdown.currentIndex()

        # if choice == 0:
        #     pass

        # elif choice == 1:
        #     print("Selected color 1 ")

        # elif choice == 2:
        #     print("Selcted color 2")
        
        # elif choice == 3:
        #     print("Selected color 3")
        return 0

    def legend(self, state):
        # if state == Qt.Checked:
        #     print("Legend On")
        # elif state == Qt.Unchecked:
        #     print("Legend Off")
        return 1

    def display_image(self, view, image_path):
        """Loads and displays an image in the specified QGraphicsView."""
        scene = QGraphicsScene()
        pixmap = QPixmap(image_path)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(pixmap_item)
        view.setScene(scene)
        