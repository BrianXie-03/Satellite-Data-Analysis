# 🛰️ Satellite Data Analysis Interface 🛰️

A user-friendly interface built for efficient **comparison and visualization of satellite data** from two different sources or time periods. 
This tool is designed to assist researchers, scientists, and analysts in quality control, trend analysis, and inter-satellite data validation.

---

## 🌐 Overview 🌐

This application interface allows users to load and compare variables from **two netCDF4-format satellite data files**, offering side-by-side and difference data comparison. 
The application is equipped with dropdowns for selecting variables, optional QC bit analysis, and multiple visualizations statistical insights.

---

## ✨ Features

- 📂 **Dual File Input**: Load two satellite netCDF4 files for comparison (currently supports .nc files, more to be implemented).
- 📊 **Variable Selection**: Dynamically populated dropdowns with multi-dimensional variables (e.g., 2D, 3D).
- ✅ **QC Flag Support**: Optional bit-level selection for QC variables (e.g., start bit and length).
- 🗺️ **Visualization Panel**: View and compare side-by-side / difference plots of the selected data.
- 📈 **Statistics Summary**: Compute and display statistics for selected variables (e.g., RMSE, Standard Deviation, Correlation).
- 🔄 **Synchronized Navigation**: Compare datasets across identical time or spatial dimensions.

---

## 🛠️ How to Use

1. **Launch the Interface**.
2. **Load Satellite Files**: Use the dropdown menus to select two netCDF4 files.
3. **Projection Type**: Select projection type (to be implemented)
4. **Select Variables**: Choose which variable to compare from each file (e.g., Ch1_REF and BRF1)
5. **(Optional)**: Check the QC box and specify the bit range if you're analyzing QC flags.
6. **Region of Interest**: Import shapefile (to be implemented), select latitude and longtitude (to be implemented), or keep general RoI.
7. **Analyze Stats**: Head to the 'Stat' panel for statistical summaries and distribution comparisons as well as map projection.
8. **Visualize**: Use the 'Visuals' tab to compare statistical plots and observe data trends.
---

```bash
├── interface.py                 # App entry point; initializes and runs the GUI
├── ui_interface.py              # Main UI class (from .ui or manual layout)
├── interface.ui                 # .ui file for integration
├── ui/
│   ├── __init__.py
│   ├── data.py                  # DataPanel - handles file input and variable selection
│   ├── stats.py                 # StatPanel - displays computed statistics
│   └── visual.py                # VisualsPanel - renders matplotlib plots
├── scripts/
│   ├── __init__.py
│   └── verification.py          # Comparison class - handles comparison logic, QC filtering
└── README.md                    # Project documentation
```
---

## 👨‍💻 Author

**Brian Xie**  
Feel free to reach out with questions or contributions!

---

© 2025 Brian Xie — All rights reserved.
