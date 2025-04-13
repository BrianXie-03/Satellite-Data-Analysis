from functools import partial
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui_interface import Ui_MainWindow 
from PyQt5.QtCore import Qt
import numpy as np
# import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import sys
import os
from ui.data import DataPanel
from ui.stats import StatPanel
from ui.visual import VisualsPanel
from scripts.verification import Comparison


class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.cmp = Comparison(self)

        # Initialize panels
        self.data_panel = DataPanel(self)
        self.stat_panel = StatPanel(self)
        self.visuals_panel = VisualsPanel(self)

        ########### Panel 1 ###########
        # creating borders for label
        self.fileLabel.setStyleSheet("border: 2px solid rgba(0, 0, 0, 0.3); padding: 5px;")
        self.fileLabel_2.setStyleSheet("border: 2px solid rgba(0, 0, 0, 0.3); padding: 5px;")

        # Initialize file paths for tracking
        self.file_path = None
        self.file_path_2 = None
        self.file_path_3 = None # only usable when shapefile is needed

        #upload button for comparison files
        self.uploadButton.clicked.connect(partial(self.upload_file_and_store, self.fileLabel, "file_path_1", self.remove))
        self.uploadButton_2.clicked.connect(partial(self.upload_file_and_store, self.fileLabel_2, "file_path_2", self.remove_2))

        # Setting up the Remove Button for comparison files (hidden if there is no file inputted)
        self.remove.clicked.connect(partial(self.data_panel.clear_file, self.fileLabel, "file_path_1", self.remove))
        self.remove.setEnabled(False)
        self.remove_2.clicked.connect(partial(self.data_panel.clear_file, self.fileLabel_2, "file_path_2", self.remove_2))
        self.remove_2.setEnabled(False)

        for i in range(self.bit_grid.count()):
            widget = self.bit_grid.itemAt(i).widget()
            if widget:
                widget.setVisible(False) 

        self.qc_check.stateChanged.connect(self.data_panel.toggle_qc)

        #ROI Dropdown
        for i in range(self.horizontalLayout_sf.count()):
            widget = self.horizontalLayout_sf.itemAt(i).widget()
            if widget:
                widget.setVisible(False) 
        self.data_panel.hide_all_widgets_in_grid()
        self.ROI_combo.currentIndexChanged.connect(self.data_panel.handle_selection)

        #Compare
        self.compare_button.clicked.connect(self.compare_files)

        ########### Panel 2 ###########

        # ensure that all graphical boxes are hidden
        self.graph_1.setVisible(False)
        self.graph_2.setVisible(False)
        self.difference_graph.setVisible(False)
        self.stat_result.setVisible(False)

        # interaction if generate is clicked
        self.generate_button.clicked.connect(self.stat_panel.dropdowns)

        ########### Panel 3 ###########
        self.generate_vis.clicked.connect(self.visuals_panel.visual_selection)

        # keeping stuff hidden for now, since no implementation
        self.visualSettings_label.setVisible(False)
        self.x_axis_input.setVisible(False)
        self.x_axis_label.setVisible(False)
        self.y_axis_input.setVisible(False)
        self.y_axis_label.setVisible(False)
        self.scheme_label.setVisible(False)
        self.scheme_dropdown.setVisible(False)
        self.legend_checkbox.setVisible(False)
        self.legend_label.setVisible(False)
        self.plotLabel_label.setVisible(False)

        # self.visualType_dropdown.currentIndexChanged.connect(self.visuals_panel.visual_selection)
        # self.scheme_dropdown.currentIndexChanged.connect(self.visuals_panel.color_schemes)
        # self.legend_checkbox.stateChanged.connect(self.visuals_panel.legend)
        #self.x,y,etc


        ########### Menu Bar ###########
        self.action_Clear.triggered.connect(self.data_panel.clear_files)
        self.action_Quit.triggered.connect(self.data_panel.close_application)

    def upload_file_and_store(self, label, path_attr, remove_button):
        # Call upload_file and capture the return value
        source_name = self.data_panel.upload_file(label, path_attr, remove_button)
        
        # Now store the file path in an instance variable
        setattr(self, path_attr, source_name)
        print(f"Stored file path: {getattr(self, path_attr)}")

    def compare_files(self):
        """Retrieves file paths and calls comparison function"""
        file1 = self.fileLabel.text()
        file2 = self.fileLabel_2.text()
        results = self.cmp.compare_brf_files(file1, file2, output_dir="results/brf_analysis", projection=self.comboBox.currentText())
        # refl, qc = results.items()
        # # _,qc = qc1
        # print(refl)
        # print('\n')
        # # qc -> quality score (some general information) -> value_stats (bit information)
        # print(qc)



def main():
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
