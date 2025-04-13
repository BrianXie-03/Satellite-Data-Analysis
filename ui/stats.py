from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QMainWindow, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
import scipy
import sys
import xarray as xr
import numpy as np
from scipy.stats import pearsonr
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt



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
        self.ui.stat_result.setVisible(True)
        data = self.calculate_statistics()
        self.ui.stat_result.setReadOnly(True)

        if choice == 0:
            pass
        elif choice == 1:
            self.ui.stat_result.setText(f"<b>RMSE:</b> {data[0]}")

        elif choice == 2:
            self.ui.stat_result.setText(f"<b>Correlation:</b> {data[1]}")        
        elif choice == 3:
            self.ui.stat_result.setText(f"<b>Standard Deviation:</b> {data[2]}")
            
        self.ui.stat_result.setStyleSheet("font-size: 16pt; font-family: Poppins;")
        self.ui.stat_result.setAlignment(Qt.AlignCenter)

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
            self.display_image(self.ui.graph_1, "/home/brian/research/results/brf_analysis/BRF_New.png")
            self.display_image(self.ui.graph_2, "/home/brian/research/results/brf_analysis/BRF_Reference.png")
            
            print("Selected Side-by-Side")

        elif choice == 2:
            self.ui.difference_graph.setVisible(True)
            self.display_image(self.ui.difference_graph, "/home/brian/research/results/brf_analysis/BRF_Difference.png")
            print("Selected Difference")

    def display_image(self, view, image_path):
        """Loads and displays an image in the specified QGraphicsView."""
        scene = QGraphicsScene()
        pixmap = QPixmap(image_path)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(pixmap_item)
        view.setScene(scene)
    
    def calculate_statistics(self):

        file1 = self.ui.fileLabel.text()
        file2 = self.ui.fileLabel_2.text()
        d1 = xr.open_dataset(file1)
        d2 = xr.open_dataset(file2)
        channel1 = str(self.ui.fileDropdown1.currentText())
        channel2 = str(self.ui.fileDropdown2.currentText())
        extract_data1 = d1[channel1][:].values.flatten()
        extract_data2 = d2[channel2][:].values.flatten()

        if self.ui.qc_check.isChecked():
            bit_start = int(self.ui.input_start_bit.text())
            bit_length = int(self.ui.input_bit_length.text())            

            d1_clean = self.extract_bits(d1["Ref_QF"][:], bit_start, bit_length)
            d2_clean = self.extract_bits(d2["DQF"][:], bit_start, bit_length)
            mask = (np.isfinite(d1_clean) & np.isfinite(d2_clean) )
            d1_clean = d1_clean[mask]
            d2_clean = d2_clean[mask]

        else:
            mask = (~np.isnan(extract_data1) & ~np.isnan(extract_data2) )
            d1_clean = extract_data1[mask]
            d2_clean = extract_data2[mask]

        ## RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((d1_clean - d2_clean) ** 2))

        ## Correlation Coefficient (Pearson r)
        correlation, _ = pearsonr(d1_clean, d2_clean)

        ## Standard Deviation of differences
        std_dev = np.std(d1_clean - d2_clean)

        return [rmse, correlation, std_dev]
    
    def extract_bits(self, qc_data, start_bit, num_bits):
        qc_int = np.nan_to_num(qc_data, nan=0).astype(np.uint8)
        mask = ((1 << num_bits) - 1) << start_bit
        extracted_bits = (qc_int & mask) >> start_bit

        result = extracted_bits.astype(float)
        result[np.isnan(qc_data)] = np.nan
        return result
