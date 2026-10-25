import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class OhioT1DMLoader:
    """Loads patient data from the OhioT1DM dataset (XML format)."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def find_patient_files(self):
        """Recursively search for XML files containing patient data."""
        search_path = os.path.join(self.data_dir, '**', '*.xml')
        files = glob.glob(search_path, recursive=True)
        if not files:
            logger.warning(f"No XML files found in {self.data_dir}. Please see dataset_setup.md")
        return files

    def extract_events(self, root, tag_name, value_attr):
        """Helper to extract time series events from an XML tag."""
        events = []
        element = root.find(tag_name)
        if element is not None:
            for ev in element.findall('event'):
                ts = ev.get('ts')
                val = ev.get(value_attr)
                if ts and val:
                    try:
                        events.append({'timestamp': pd.to_datetime(ts, dayfirst=True),
                                       tag_name: float(val)})
                    except (ValueError, Exception):
                        pass
        return pd.DataFrame(events)

    def load_patient(self, file_path: str) -> dict:
        """Parse a single patient XML file into dataframes."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            patient_id = root.get('id', 'unknown')
            df_glucose = self.extract_events(root, 'glucose_level', 'value')
            df_bolus = self.extract_events(root, 'bolus', 'dose')
            df_meal = self.extract_events(root, 'meal', 'carbs')
            return {
                'patient_id': patient_id,
                'glucose': df_glucose,
                'insulin': df_bolus,
                'carbs': df_meal,
                'file_path': file_path
            }
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return None

    def load_all_patients(self) -> dict:
        """Load all available patients into a dictionary."""
        files = self.find_patient_files()
        patients = {}
        for f in files:
            p_data = self.load_patient(f)
            if p_data and not p_data['glucose'].empty:
                pid = p_data['patient_id']
                if pid not in patients:
                    patients[pid] = []
                patients[pid].append(p_data)

        merged_patients = {}
        for pid, data_list in patients.items():
            glucose_frames = [d['glucose'] for d in data_list if not d['glucose'].empty]
            insulin_frames = [d['insulin'] for d in data_list if not d['insulin'].empty]
            carbs_frames = [d['carbs'] for d in data_list if not d['carbs'].empty]

            mg = pd.concat(glucose_frames).drop_duplicates('timestamp').sort_values('timestamp') if glucose_frames else pd.DataFrame()
            mi = pd.concat(insulin_frames).drop_duplicates('timestamp').sort_values('timestamp') if insulin_frames else pd.DataFrame()
            mc = pd.concat(carbs_frames).drop_duplicates('timestamp').sort_values('timestamp') if carbs_frames else pd.DataFrame()

            merged_patients[pid] = {
                'patient_id': pid,
                'glucose': mg,
                'insulin': mi,
                'carbs': mc
            }
        return merged_patients
