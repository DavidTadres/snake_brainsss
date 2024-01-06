
from xml.etree import ElementTree as ET

def extract_resolution(metadata_path):
    """
    Extract resolution from metadata file
    :param metadata_path:
    :return:
    """
    tree = ET.parse(metadata_path)
    root = tree.getroot()
    # root[1] is PVStateShard
    for current_entry in root[1]:
        if current_entry.get('key') == 'micronsPerPixel':
            for current_axis in current_entry:
                if current_axis.get('index') == 'XAxis':
                    temp_x = current_axis.get('value')
                elif current_axis.get('index') == 'YAxis':
                    temp_y = current_axis.get('value')
                elif current_axis.get('index') == 'ZAxis':
                    temp_z = current_axis.get('value')
    resolution = (round(float(temp_x, 5)),
                  round(float(temp_y, 5)),
                  round(float(temp_z, 5))
                  )

    return(resolution)