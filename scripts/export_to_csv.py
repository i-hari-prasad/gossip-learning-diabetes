import os
import xml.etree.ElementTree as ET
import pandas as pd

def xml_to_csv(xml_file, output_dir):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        patient_id = root.attrib.get('id', 'unknown')
        filename = os.path.basename(xml_file).replace('.xml', '')
        
        data_records = []
        
        for child in root:
            event_type = child.tag
            for event in child:
                record = {
                    'patient_id': patient_id,
                    'timestamp': event.attrib.get('ts'),
                    'event_type': event_type,
                    'value': event.attrib.get('value', ''),
                    'carbs': event.attrib.get('carbs', ''),
                    'dose': event.attrib.get('dose', '')
                }
                data_records.append(record)
                
        if not data_records:
            return
            
        df = pd.DataFrame(data_records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        df = df.sort_values('timestamp')
        
        out_path = os.path.join(output_dir, f"{filename}.csv")
        df.to_csv(out_path, index=False)
        print(f"Exported {out_path}")
        
    except Exception as e:
        print(f"Error processing {xml_file}: {e}")

if __name__ == '__main__':
    data_dir = 'data/synthetic'
    out_dir = 'data/csv_exports'
    os.makedirs(out_dir, exist_ok=True)
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.xml'):
                xml_to_csv(os.path.join(data_dir, file), out_dir)
    else:
        print(f"Directory {data_dir} not found.")
