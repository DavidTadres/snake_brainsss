"""
I've started adding scan.json data to master_2p.csv as I accidentally imaged at 800nm one time and almost didn't notice.
This will help anyone in the future avoid this potential pitfall!

Since I have some data I preprocessed before adding this newest addition to the csv, this script manually adds it.
"""

import pathlib
import pandas as pd
import numpy as np
import json
from xml.etree import ElementTree as ET
from lxml import etree, objectify

csv_path = pathlib.Path('\\\\oak-smb-trc.stanford.edu\\groups\\trc\\data\\David\\Bruker\\preprocessed\\SS84990_DNa03_x_GCaMP6f\\master_2P.csv')
csv_file = pd.read_csv(csv_path, index_col=0)

missing_columns = ['date', 'time', 'laser_power', 'laser_wavelength', 'y_dim',
                   'x_voxel_size', 'y_voxel_size', 'z_voxel_size', 'x_dim', 'PMT 1 HV',
                   'PMT 2 HV', 'PMT 3 HV', 'z_dim'
                   ]

# checks if 'z_dim' is NaN or not
rows_with_missing_scan_json = np.where(np.isnan(csv_file[missing_columns[-1]]))[0]

def load_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data

def create_imaging_json(xml_source_file):
    # Make empty dict
    source_data = {}

    # Get datetime
    try:
        datetime_str, _, _ = get_datetime_from_xml(xml_source_file)
    except:
        print("No xml or cannot read.")
        ##sys.stdout.flush()
        return
    date = datetime_str.split("-")[0]
    time = datetime_str.split("-")[1]
    source_data["date"] = str(date)
    source_data["time"] = str(time)

    # Get rest of data
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
            # This is not great - this is just the first pockels value
            indices = statevalue.findall("IndexedValue")
            laser_power_overall = int(float(indices[0].get("value")))
            source_data["laser_power"] = laser_power_overall
        if key == "laserWavelength":
            index = statevalue.findall("IndexedValue")
            laser_wavelength = int(float(index[0].get("value")))
            source_data["laser_wavelength"] = laser_wavelength
        if key == "pmtGain":
            indices = statevalue.findall("IndexedValue")
            for index in indices:
                index_num = index.get("index")
                # I changed this from 'red' and 'green' to the actual description used by the
                # microscope itself! Since we now have 2 Brukers, this seems safer!
                if index_num == "0":
                    source_data[index.get("description")] = int(float(index.get("value")))
                if index_num == "1":
                    source_data[index.get("description")] = int(float(index.get("value")))
                if index_num == "2":
                    source_data[index.get("description")] = int(float(index.get("value")))
        if key == "pixelsPerLine":
            source_data["x_dim"] = int(float(statevalue.get("value")))
        if key == "linesPerFrame":
            source_data["y_dim"] = int(float(statevalue.get("value")))
    sequence = source.findall("Sequence")[0]
    last_frame = sequence.findall("Frame")[-1]
    source_data["z_dim"] = int(last_frame.get("index"))

    # Save data
    # with open(os.path.join(os.path.split(xml_source_file)[0], 'scan.json'), 'w') as f:
    with open(pathlib.Path(xml_source_file.parent, "scan.json"), "w") as f:
        json.dump(source_data, f, indent=4)

def get_datetime_from_xml(xml_file):
    ##print('Getting datetime from {}'.format(xml_file))
    ##sys.stdout.flush()
    tree = ET.parse(xml_file)
    root = tree.getroot()
    datetime = root.get("date")
    # will look like "4/2/2019 4:16:03 PM" to start

    # Get dates
    date = datetime.split(" ")[0]
    month = date.split("/")[0]
    day = date.split("/")[1]
    year = date.split("/")[2]

    # Get times
    time = datetime.split(" ")[1]
    hour = time.split(":")[0]
    minute = time.split(":")[1]
    second = time.split(":")[2]

    # Convert from 12 to 24 hour time
    am_pm = datetime.split(" ")[-1]
    if am_pm == "AM" and hour == "12":
        hour = str(00)
    elif am_pm == "AM":
        pass
    elif am_pm == "PM" and hour == "12":
        pass
    else:
        hour = str(int(hour) + 12)

    # Add zeros if needed
    if len(month) == 1:
        month = "0" + month
    if len(day) == 1:
        day = "0" + day
    if len(hour) == 1:
        hour = "0" + hour

    # Combine
    datetime_str = year + month + day + "-" + hour + minute + second
    datetime_int = int(year + month + day + hour + minute + second)
    datetime_dict = {
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
    }

    return datetime_str, datetime_int, datetime_dict

for current_row in rows_with_missing_scan_json:
    print(current_row)

    # folder where we need to create read scan.json
    current_path = csv_file['Dataset folder'][current_row] # i.e. '/oak/stanford/groups/trc/data/David/Bruker/preprocessed/SS84990_DNa03_x_GCaMP6f/fly_001'
    current_subfolder = csv_file['Expt ID'][current_row] # i.e. 'func0'

    try:
        # csv_path.parent() # i.e. WindowsPath('//oak-smb-trc.stanford.edu/groups/trc/data/David/Bruker/preprocessed/SS84990_DNa03_x_GCaMP6f')
        scan_json_path = pathlib.Path(csv_path.parent, pathlib.Path(current_path).name, current_subfolder, "imaging/scan.json")
        scan_json = load_json(scan_json_path)
        print("Successfully loaded scan.json file")

        print(scan_json['laser_wavelength']) # This tests if a new scan.json file exists. Will fail if not and
        # in except will create new json.
    except:
        # re-create the scan.json file from scratch as it's different now than when it was created (i.e. we now
        # report laser wavelength).
        create_imaging_json(pathlib.Path(csv_path.parent, pathlib.Path(current_path).name,
                                         current_subfolder, "imaging/recording_metadata.xml"))

    for current_column in missing_columns:
        csv_file.loc[current_row, current_column] = scan_json[current_column]
        #csv_file[current_column][current_row] = scan_json[current_column]

# Save updated master.csv file
csv_file.to_csv(csv_path)
