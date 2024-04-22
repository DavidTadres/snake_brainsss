from xml.etree import ElementTree as ET
from lxml import etree, objectify
from pathlib import Path

source_data = {}

xml_source_file = Path('/Volumes/groups/trc/data/David/Bruker/imports/20240405/fly1/anat0/TSeries-12172018-1322-002/TSeries-12172018-1322-002.xml')

tree = objectify.parse(xml_source_file)

source = tree.getroot()
statevalues = source.findall("PVStateShard")[0].findall("PVStateValue")

for statevalue in statevalues:
    key = statevalue.get("key")
    if key == "micronsPerPixel":
        indices = statevalue.findall("IndexedValue")
        for index in indices:
            axis = index.get("index")
            if axis == "XAxis":
                source_data["x_voxel_size"] = float(index.get("value"))
            elif axis == "YAxis":
                source_data["y_voxel_size"] = float(index.get("value"))
            elif axis == "ZAxis":
                source_data["z_voxel_size"] = float(index.get("value"))
    if key == "laserPower":
        break
        # This is not great - this is just the first pockels value
        indices = statevalue.findall("IndexedValue")
        laser_power_overall = int(float(indices[0].get("value")))
        source_data["laser_power"] = laser_power_overall
    if key == "laserWavelength":

        index = statevalue.findall("IndexedValue index")
        laser_wavelength = int(float(index[0]).get("value"))
        source_data["laser_wavelength"] = laser_wavelength
    if key == "pmtGain":
        indices = statevalue.findall("IndexedValue")
        for index in indices:
            index_num = index.get("index")
            if index_num == "0":
                source_data["PMT_red"] = int(float(index.get("value")))
            if index_num == "1":
                source_data["PMT_green"] = int(float(index.get("value")))
    if key == "pixelsPerLine":
        source_data["x_dim"] = int(float(statevalue.get("value")))
    if key == "linesPerFrame":
        source_data["y_dim"] = int(float(statevalue.get("value")))
