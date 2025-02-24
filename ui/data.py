from functools import partial
from pathlib import Path
from netCDF4 import Dataset
from PyQt5.QtWidgets import QApplication,QFileDialog

class DataPanel():
    def __init__(self, ui):
        self.ui = ui  # Reference to the main UI

    def hide_all_widgets_in_grid(self):
        for i in range(self.ui.GeoGrid.count()):
            item = self.ui.GeoGrid.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setVisible(False)
                else:
                    layout = item.layout()
                    if layout:
                        self.hide_widgets_in_layout(layout)

    def hide_widgets_in_layout(self, layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item:
                widget = item.widget() 
                if widget:
                    widget.setVisible(False)
                else:
                    nested_layout = item.layout()
                    if nested_layout:
                        self.hide_widgets_in_layout(nested_layout)

    def show_all_widgets_in_grid(self):
        for i in range(self.ui.GeoGrid.count()):
            item = self.ui.GeoGrid.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setVisible(True)
                else:
                    layout = item.layout()
                    if layout:
                        self.show_widgets_in_layout(layout)

    def show_widgets_in_layout(self, layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item:
                widget = item.widget() 
                if widget:
                    widget.setVisible(True)
                else:
                    nested_layout = item.layout()
                    if nested_layout:
                        self.show_widgets_in_layout(nested_layout)

    def toggle_qc(self):
        if self.ui.qc_check.isChecked():
            for i in range(self.ui.bit_grid.count()):
                widget = self.ui.bit_grid.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)
        else:
            for i in range(self.ui.bit_grid.count()):
                widget = self.ui.bit_grid.itemAt(i).widget()
                if widget:
                    widget.setVisible(False)

    def handle_selection(self):
        choice = self.ui.ROI_combo.currentIndex()
        print(choice)

        #shapefile prompt
        for i in range(self.ui.horizontalLayout_sf.count()):
            widget = self.ui.horizontalLayout_sf.itemAt(i).widget()
            if widget:
                widget.setVisible(False)

        self.hide_all_widgets_in_grid()

        if choice == 0:
            pass

        elif choice == 1:
            for i in range(self.ui.horizontalLayout_sf.count()):
                widget = self.ui.horizontalLayout_sf.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)
            
            self.ui.fileLabel_sf.setStyleSheet("border: 2px solid rgba(0, 0, 0, 0.3); padding: 5px;")
            self.ui.uploadButton_sf.clicked.connect(partial(self.upload_file, self.fileLabel, "file_path_3", self.remove))
            self.ui.remove_sf.clicked.connect(partial(self.clear_file, self.fileLabel, "file_path_3", self.remove))
            self.ui.remove_sf.setEnabled(False)

            print("Shapefile selected")

        elif choice == 2:
            self.show_all_widgets_in_grid()
            print("Lat/Long selected")

    def apply_epsg(self):
        epsg_code = self.ui.EPSG_value.text()

        #Check to see if the epsg code is valid
        try:
            epsg_code = int(epsg_code)
            self.ui.EPSG_warning.setText(f"Setting EPSG Code to {epsg_code}")
        except ValueError:
            self.ui.EPSG_warning.setText("Invalid EPSG Code. Please enter a valid integer.")

    def clear_files(self):
        self.clear_file(self.ui.fileLabel, "file_path", self.ui.remove)
        self.clear_file(self.ui.fileLabel_2, "file_path_2", self.ui.remove_2)
        print("Both files cleared.")

    def close_application(self):
        QApplication.quit()

    def clear_file(self, label, path_attr, remove_button):
        label.setText("No file selected")
        setattr(self, path_attr, None)  # Reset stored file path
        remove_button.setEnabled(False)  # Disable remove button
        print("File removed")

    def upload_shapefile(self, label, path_attr, remove_button):
        file_name, _ = QFileDialog.getOpenFileName(self.ui, "Open File", "", "All Files (*);;Shapefile (*.shp)")

        if file_name:
            source_name = Path(file_name).name
            label.setText(f"{source_name}")
            setattr(self, path_attr, file_name)
            print(f"File selected: {file_name}")
            self.process_file(file_name)
            remove_button.setEnabled(True)

    def upload_file(self, label, path_attr, remove_button):
        file_name, _ = QFileDialog.getOpenFileName(self.ui, "Open File", "", "All Files (*);;NetCDF (*.nc);;GeoTIFF (*.tif);;HDF (*.h5);;CSV (*.csv)")

        if file_name:
            source_name = Path(file_name)
            label.setText(f"{source_name}")
            setattr(self, path_attr, file_name)
            print(f"File selected: {file_name}")
            
            if path_attr == "file_path_1":
                self.populate_dropdown(self.ui.fileDropdown1, file_name)
            elif path_attr == "file_path_2":
                self.populate_dropdown(self.ui.fileDropdown2, file_name)
            remove_button.setEnabled(True)
            return file_name
        return None
    
    def populate_dropdown(self, dropdown, filename):
        try:
            nc = Dataset(filename, 'r')
            variables = [var for var in nc.variables.keys() if len(nc.variables[var].dimensions) >= 2]
            dropdown.clear()
            dropdown.addItems(variables)
            nc.close()
        except Exception as e:
            print(f"Error loading netCDF file: {e}")