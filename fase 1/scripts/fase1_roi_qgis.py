# ============================================================
# fase1_roi_qgis.py
# Phase 1 – ROI visualization in QGIS
# Run from: Plugins → Python Console → Open Editor → Run
# ============================================================

from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry,
                       QgsPointXY, QgsProject, QgsFillSymbol)
from qgis.utils import iface

# ROI polygon in WGS84 (EPSG:4326)
coords = [
    (-74.925728, 5.086328),
    (-74.834747, 5.086328),
    (-74.834747, 4.997067),
    (-74.925728, 4.997067),
    (-74.925728, 5.086328)
]

# Create memory layer
layer = QgsVectorLayer("Polygon?crs=EPSG:4326", "ROI_Armero", "memory")
provider = layer.dataProvider()

# Create polygon feature
feature = QgsFeature()
points = [QgsPointXY(lon, lat) for lon, lat in coords]
feature.setGeometry(QgsGeometry.fromPolygonXY([points]))
provider.addFeature(feature)
layer.updateExtents()

# Style: transparent fill + red border
symbol = QgsFillSymbol.createSimple({
    'color': '231,76,60,40',
    'outline_color': '231,76,60,255',
    'outline_width': '1.2',
    'outline_style': 'solid'
})
layer.renderer().setSymbol(symbol)
layer.triggerRepaint()

# Add to project and zoom
QgsProject.instance().addMapLayer(layer)
iface.setActiveLayer(layer)
iface.zoomToActiveLayer()

print("ROI layer added successfully!")
