from PyQt5.QtCore import Qt


class VisualsPanel:
    def __init__(self, ui):
        self.ui = ui  # Reference to the main UI
    
    def visual_selection(self):
        choice = self.ui.visualType_dropdown.currentIndex()

        if choice == 0:
            pass

        elif choice == 1:
            print("Selected Scatter Plot")

        elif choice == 2:
            print("Selcted Line Graph")
        
        elif choice == 3:
            print("Selected Difference Map")

        elif choice == 4:
            print("Selected Histogram")

    def color_schemes(self):
        choice = self.ui.scheme_dropdown.currentIndex()

        if choice == 0:
            pass

        elif choice == 1:
            print("Selected color 1 ")

        elif choice == 2:
            print("Selcted color 2")
        
        elif choice == 3:
            print("Selected color 3")

    def legend(self, state):
        if state == Qt.Checked:
            print("Legend On")
        elif state == Qt.Unchecked:
            print("Legend Off")
